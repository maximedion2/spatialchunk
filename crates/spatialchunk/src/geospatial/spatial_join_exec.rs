// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::fmt::Formatter;
use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;
use datafusion::common::{project_schema, DataFusionError, JoinType, Result};
use datafusion::config::ConfigOptions;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::equivalence::{join_equivalence_properties, ProjectionMapping};
use datafusion::physical_expr::expressions::{lit, DynamicFilterPhysicalExpr};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, ChildPushdownResult, FilterDescription, FilterPushdownPhase,
    FilterPushdownPropagation,
};
use datafusion::physical_plan::joins::utils::{
    adjust_right_output_partitioning, build_join_schema, check_join_is_valid, ColumnIndex,
    JoinFilter,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::future::{BoxFuture, FutureExt};

use crate::geospatial::geos::{build_from_streams, BBoxedGeoStream, IndexedBuildSide};
use crate::geospatial::join_predicates::{RelationPredicate, SpatialRelationType};
use crate::geospatial::probe_pruning_expr::ProbePruningExpr;
use crate::geospatial::spatial_join_stream::{SharedIndexFuture, SpatialJoinStream};
use crate::zarr_store_opener::zarr_store_interface::SpatialPruningExpr;

#[derive(Debug, Clone)]
pub struct SpatialJoinExec {
    pub(crate) left: Arc<dyn ExecutionPlan>,
    pub(crate) right: Arc<dyn ExecutionPlan>,
    pub(crate) predicate: RelationPredicate,
    pub(crate) filter: Option<JoinFilter>,
    pub(crate) join_type: JoinType,
    join_schema: SchemaRef,
    column_indices: Vec<ColumnIndex>,
    projection: Option<Vec<usize>>,
    props: Arc<PlanProperties>,

    // Lazily initialized on the first `execute()` call, then shared
    // across all probe partitions, and of of the partitions will
    // await it while the others just wait for the result that will
    // get cached in the shared future.
    build_side_shared: Arc<Mutex<Option<SharedIndexFuture>>>,

    // Probe-side chunk-pruning filter, shared with the probe child. Set during
    // filter pushdown (`handle_child_pushdown_result`) when the child accepts
    // it; filled in with the real spatial index once the build side is ready.
    // None when pushdown didn't happen or the child declined.
    dynamic_filter: Option<Arc<DynamicFilterPhysicalExpr>>,
}

impl SpatialJoinExec {
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        predicate: RelationPredicate,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        match join_type {
            JoinType::Inner | JoinType::Left | JoinType::Right => {}
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "BulkSpatialJoinExec does not support join type {other:?}"
                )));
            }
        }

        // Build the schema and column indices for the final record batch.
        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &[])?;
        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &join_type);
        let join_schema = Arc::new(join_schema);

        let props = Self::compute_properties(
            &left,
            &right,
            &join_type,
            Arc::clone(&join_schema),
            projection.as_ref(),
        )?;

        // The project is kept because when calling new_with_chilldren,
        // the projection needs to be applied again.
        Ok(Self {
            left,
            right,
            predicate,
            filter,
            join_type,
            join_schema,
            column_indices,
            projection,
            props: Arc::new(props),
            build_side_shared: Arc::new(Mutex::new(None)),
            dynamic_filter: None,
        })
    }

    // Returns a copy of this exec carrying the pushed-down dynamic filter. Used
    // by handle_child_pushdown_result to attach the filter once the probe
    // child accepts it.
    fn with_dynamic_filter(&self, dynamic_filter: Arc<DynamicFilterPhysicalExpr>) -> Result<Self> {
        // A plain clone only carries over immutable config, not execution state.
        // That's correct only while the build side is still uninitialized, which
        // holds because pushdown runs before execute.
        let build_side_unset = self
            .build_side_shared
            .lock()
            .map_err(|e| DataFusionError::Internal(e.to_string()))?
            .is_none();
        if !build_side_unset {
            return Err(DataFusionError::Internal(
                "with_dynamic_filter called after the build side was initialized".into(),
            ));
        }
        let mut new = self.clone();
        new.dynamic_filter = Some(dynamic_filter);
        Ok(new)
    }

    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        join_type: &JoinType,
        join_schema: SchemaRef,
        projection: Option<&Vec<usize>>,
    ) -> Result<PlanProperties> {
        // Inherit the equivalence that might already exists,
        // but the spatial join does introduce any new equivalence.
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            join_type,
            Arc::clone(&join_schema),
            &[false, false],
            None,
            &[],
        )?;

        // inner or right join -> output partitioning matches the
        // probe (right) side, the adjustment is to shift column
        // indices by the number of column on the left side if
        // there is a hash property for the probe partitions.
        //
        // left join -> output partitioning has the same number
        // of partitions as right side, but the last partition
        // to emit includes unvisited build side rows, so the hash
        // properties is not preserved (if there is one).
        let mut output_partitioning = match join_type {
            JoinType::Inner | JoinType::Right => adjust_right_output_partitioning(
                right.output_partitioning(),
                left.schema().fields().len(),
            )?,
            JoinType::Left => {
                Partitioning::UnknownPartitioning(right.output_partitioning().partition_count())
            }
            _ => unreachable!(),
        };

        // Ff left is unbounded, since you need to build the spatial
        // index first, the plan doesn't really make sense, but
        // technically you can say it's a Final emission, though in
        // practice you would just consume left side batches forever.
        //
        // Other than that, you just get the probe side emission type,
        // except that for a left join, the unvisited rows are emitted
        // once at the very end, with the last batch of the last partition,
        // hence the Both case.
        let emission_type = if left.boundedness().is_unbounded() {
            EmissionType::Final
        } else if right.pipeline_behavior() == EmissionType::Incremental {
            match join_type {
                JoinType::Inner | JoinType::Right => EmissionType::Incremental,
                JoinType::Left => EmissionType::Both,
                _ => unreachable!(),
            }
        } else {
            right.pipeline_behavior()
        };

        // Basically inherit the most "unbounded" from left and right.
        // unbounded (infinite) > unbounded (finite) > bounded.
        let lb = left.boundedness();
        let rb = right.boundedness();
        let boundedness = if matches!(
            lb,
            Boundedness::Unbounded {
                requires_infinite_memory: true
            }
        ) || matches!(
            rb,
            Boundedness::Unbounded {
                requires_infinite_memory: true
            }
        ) {
            Boundedness::Unbounded {
                requires_infinite_memory: true,
            }
        } else if matches!(lb, Boundedness::Unbounded { .. })
            || matches!(rb, Boundedness::Unbounded { .. })
        {
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            }
        } else {
            Boundedness::Bounded
        };

        // The projection can modify the paritioning and the
        // equivalence because it can drop columns that the partitions
        // were hashed on or that were equivalent, or it can shift the
        // index of those that stay after other columns were dropped.
        if let Some(projection) = projection {
            let projection_mapping = ProjectionMapping::from_indices(projection, &join_schema)?;
            let out_schema = project_schema(&join_schema, Some(projection))?;
            output_partitioning = output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            emission_type,
            boundedness,
        ))
    }
}

impl DisplayAs for SpatialJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let p = &self.predicate;
                let display_on = format!(", on={:?}({}, {})", p.relation_type, p.left, p.right);
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projection = self.projection.as_ref().map_or_else(
                    || "".to_string(),
                    |proj| {
                        format!(
                            ", projection=[{}]",
                            proj.iter()
                                .map(|i| format!(
                                    "{}@{}",
                                    self.join_schema.fields().get(*i).unwrap().name(),
                                    i
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    },
                );
                write!(
                    f,
                    "SpatialJoinExec: join_type={:?}{}{}{}",
                    self.join_type, display_on, display_filter, display_projection
                )
            }
            DisplayFormatType::TreeRender => {
                if self.join_type != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for SpatialJoinExec {
    fn name(&self) -> &str {
        "BulkSpatialJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            children[0].clone(),
            children[1].clone(),
            self.predicate.clone(),
            self.filter.clone(),
            self.join_type,
            self.projection.clone(),
        )?))
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        // A spatial join can't route arbitrary parent predicates across itself,
        // so neither child accepts them.
        let left_child = ChildFilterDescription::all_unsupported(&parent_filters);
        let mut right_child = ChildFilterDescription::all_unsupported(&parent_filters);

        // Honor both the master switch and the join-specific one, the master is
        // only guaranteed to override the specific flag when set via the string
        // API, so check both.
        let dynamic_filters_enabled = config.optimizer.enable_dynamic_filter_pushdown
            && config.optimizer.enable_join_dynamic_filter_pushdown;

        // Dynamic filters are a post-optimization, only push one when
        // pruning the probe (right) side is sound: Inner/Left. Right/Full must
        // emit unmatched probe rows, so their chunks can't be pruned. The filter
        // is filled in once the build side is ready.
        if dynamic_filters_enabled
            && matches!(phase, FilterPushdownPhase::Post)
            && matches!(self.join_type, JoinType::Inner | JoinType::Left)
        {
            let children = collect_columns(&self.predicate.right)
                .into_iter()
                .map(|c| Arc::new(c) as Arc<dyn PhysicalExpr>)
                .collect();
            let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(children, lit(true)));
            right_child = right_child.with_self_filter(dynamic_filter);
        }

        Ok(FilterDescription::new()
            .with_child(left_child)
            .with_child(right_child))
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let mut result = FilterPushdownPropagation::if_any(child_pushdown_result.clone());

        // Recover the same dynamic filter we pushed to the right child (index 1).
        // If the child kept it, store it on a new node so we can fill it in at
        // runtime (the child holds a clone of the same shared handle).
        if let Some(filter) = child_pushdown_result
            .self_filters
            .get(1)
            .and_then(|filters| filters.first())
        {
            let predicate = Arc::clone(&filter.predicate);
            if let Ok(dynamic_filter) = Arc::downcast::<DynamicFilterPhysicalExpr>(predicate) {
                let new_node = Arc::new(self.with_dynamic_filter(dynamic_filter)?);
                result = result.with_updated_node(new_node as Arc<dyn ExecutionPlan>);
            }
        }

        Ok(result)
    }

    // The main method, produces streams that will produce record
    // batches. Not much here, the main logic is to drain all
    // the build side streams to build the indexed build side,
    // which is done in parallel (see build_from_streams) in a future
    // that gets awaited when the probe streams run.
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let build_side_fut = {
            let mut guard = self
                .build_side_shared
                .lock()
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            if guard.is_none() {
                let left = Arc::clone(&self.left);
                let geo_expr = Arc::clone(&self.predicate.left);
                let join_type = self.join_type;
                let probe_partition_count = self.right.output_partitioning().partition_count();
                let ctx = Arc::clone(&context);

                // The probe-side geometry expression and the shared dynamic
                // filter, so the build future can publish the chunk-pruning
                // predicate into the filter once the index is ready.
                let probe_geo_expr = Arc::clone(&self.predicate.right);
                let dynamic_filter = self.dynamic_filter.clone();

                // For ST_DWithin the probe chunk bboxes must be inflated by the
                // distance threshold so the chunk prune matches what the join does.
                let probe_side_buffer = match &self.predicate.relation_type {
                    SpatialRelationType::DWithin { distance } => Some(*distance),
                    _ => None,
                };
                let is_geography = self.predicate.is_geography;

                let fut: BoxFuture<'static, Result<Arc<IndexedBuildSide>, Arc<DataFusionError>>> =
                    Box::pin(async move {
                        let num_build_partitions = left.output_partitioning().partition_count();
                        let needs_visited = matches!(join_type, JoinType::Left);

                        let streams: Vec<BBoxedGeoStream> = (0..num_build_partitions)
                            .map(|k| {
                                let stream = left.execute(k, Arc::clone(&ctx)).map_err(Arc::new)?;
                                Ok(BBoxedGeoStream::new(stream, Arc::clone(&geo_expr), None))
                            })
                            .collect::<Result<_, Arc<DataFusionError>>>()?;

                        let build_side = Arc::new(
                            build_from_streams(
                                streams,
                                needs_visited,
                                probe_partition_count,
                                is_geography,
                            )
                            .await
                            .map_err(Arc::new)?,
                        );

                        // Now that the index exists, fill in the pushed-down
                        // dynamic filter so the probe scan can prune chunks.
                        if let Some(dynamic_filter) = &dynamic_filter {
                            // the probe pruning bounds each column's values, it
                            // doesn't pair them, so no broadcast.
                            let pruning: Arc<dyn PhysicalExpr> = Arc::new(ProbePruningExpr::new(
                                Arc::clone(&probe_geo_expr),
                                Arc::clone(&build_side),
                                probe_side_buffer,
                            ));
                            let pruning = Arc::new(SpatialPruningExpr::new(pruning, false));
                            dynamic_filter.update(pruning).map_err(Arc::new)?;
                            dynamic_filter.mark_complete();
                        }

                        Ok(build_side)
                    });
                *guard = Some(fut.shared());
            }
            guard
                .as_ref()
                .ok_or_else(|| {
                    DataFusionError::Internal("build side future not initialized".into())
                })?
                .clone()
        };

        let column_indices = match &self.projection {
            Some(p) => p.iter().map(|&i| self.column_indices[i].clone()).collect(),
            None => self.column_indices.clone(),
        };

        let probe_stream = self.right.execute(partition, context)?;

        Ok(Box::pin(SpatialJoinStream::new(
            self.schema(),
            self.join_type,
            self.filter.clone(),
            probe_stream,
            column_indices,
            build_side_fut,
            self.predicate.clone(),
        )))
    }
}

#[cfg(test)]
mod exec_tests {
    use std::sync::Arc;

    use arrow_array::{BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::assert_batches_sorted_eq;
    use datafusion::common::JoinType;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::joins::utils::JoinFilter;
    use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, PhysicalExpr};
    use futures::TryStreamExt;

    use super::SpatialJoinExec;
    use crate::geospatial::join_predicates::{RelationPredicate, SpatialRelationType};
    use crate::geospatial::test_utils::{col1_gte_col2_filter, wkt_to_wkb};
    use crate::geospatial::common::wkb::{geometry_field, GeoMetadata};

    // Build side: 8 unit squares, col_1 = row index.
    //   build[1] = (1,1)-(2,2)     → within probe[0]
    //   build[3] = (6,6)-(7,7)     → within probe[5] and probe[6]
    //   build[5] = (21,21)-(22,22) → within probe[2]
    //   all others: far-away, no match
    const BUILD_WKTS: &[&str] = &[
        "POLYGON((100 100, 101 100, 101 101, 100 101, 100 100))",
        "POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))",
        "POLYGON((200 200, 201 200, 201 201, 200 201, 200 200))",
        "POLYGON((6 6, 7 6, 7 7, 6 7, 6 6))",
        "POLYGON((300 300, 301 300, 301 301, 300 301, 300 300))",
        "POLYGON((21 21, 22 21, 22 22, 21 22, 21 21))",
        "POLYGON((400 400, 401 400, 401 401, 400 401, 400 400))",
        "POLYGON((500 500, 501 500, 501 501, 500 501, 500 500))",
    ];

    // Probe side: larger squares, col_2 = row index.
    //   probe[0] = (0,0)-(5,5)     → contains build[1]
    //   probe[2] = (20,20)-(25,25) → contains build[5]
    //   probe[5] = (5,5)-(10,10)   → contains build[3]
    //   probe[6] = (4,4)-(12,12)   → contains build[3]
    //   all others: no match
    const PROBE_WKTS: &[&str] = &[
        "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))",
        "POLYGON((150 150, 160 150, 160 160, 150 160, 150 150))",
        "POLYGON((20 20, 25 20, 25 25, 20 25, 20 20))",
        "POLYGON((250 250, 260 250, 260 260, 250 260, 250 250))",
        "POLYGON((350 350, 360 350, 360 360, 350 360, 350 350))",
        "POLYGON((5 5, 10 5, 10 10, 5 10, 5 5))",
        "POLYGON((4 4, 12 4, 12 12, 4 12, 4 4))",
        "POLYGON((450 450, 460 450, 460 460, 450 460, 450 450))",
    ];

    fn build_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            geometry_field(
                "geometry_1",
                DataType::Binary,
                true,
                &GeoMetadata::default(),
            )
            .as_ref()
            .clone(),
            Field::new("col_1", DataType::Int32, true),
        ]))
    }

    fn probe_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            geometry_field(
                "geometry_2",
                DataType::Binary,
                true,
                &GeoMetadata::default(),
            )
            .as_ref()
            .clone(),
            Field::new("col_2", DataType::Int32, true),
        ]))
    }

    fn make_batch(wkts: &[&str], col_values: &[i32], geo_col: &str, val_col: &str) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            geometry_field(geo_col, DataType::Binary, true, &GeoMetadata::default())
                .as_ref()
                .clone(),
            Field::new(val_col, DataType::Int32, true),
        ]));
        let wkbs: Vec<Vec<u8>> = wkts.iter().map(|wkt| wkt_to_wkb(wkt)).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|b| b.as_slice()),
                )),
                Arc::new(Int32Array::from(col_values.to_vec())),
            ],
        )
        .unwrap()
    }

    fn make_build_batches() -> Vec<Vec<RecordBatch>> {
        vec![
            vec![
                make_batch(&BUILD_WKTS[0..2], &[0, 1], "geometry_1", "col_1"),
                make_batch(&BUILD_WKTS[2..4], &[2, 3], "geometry_1", "col_1"),
            ],
            vec![
                make_batch(&BUILD_WKTS[4..6], &[4, 5], "geometry_1", "col_1"),
                make_batch(&BUILD_WKTS[6..8], &[6, 7], "geometry_1", "col_1"),
            ],
        ]
    }

    fn make_probe_batches() -> Vec<Vec<RecordBatch>> {
        vec![
            vec![
                make_batch(&PROBE_WKTS[0..2], &[0, 1], "geometry_2", "col_2"),
                make_batch(&PROBE_WKTS[2..4], &[2, 3], "geometry_2", "col_2"),
            ],
            vec![
                make_batch(&PROBE_WKTS[4..6], &[4, 5], "geometry_2", "col_2"),
                make_batch(&PROBE_WKTS[6..8], &[6, 7], "geometry_2", "col_2"),
            ],
        ]
    }

    fn make_exec(
        join_type: JoinType,
        filter: Option<JoinFilter>,
        relation_type: SpatialRelationType,
    ) -> SpatialJoinExec {
        let build_exec =
            MemorySourceConfig::try_new_exec(&make_build_batches(), build_schema(), None).unwrap();
        let probe_exec =
            MemorySourceConfig::try_new_exec(&make_probe_batches(), probe_schema(), None).unwrap();
        let predicate = RelationPredicate::new(
            Arc::new(Column::new("geometry_1", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("geometry_2", 0)) as Arc<dyn PhysicalExpr>,
            relation_type,
        );
        // projection [1, 3]: geometry_1(0), col_1(1), geometry_2(2), col_2(3)
        SpatialJoinExec::try_new(
            build_exec,
            probe_exec,
            predicate,
            filter,
            join_type,
            Some(vec![1, 3]),
        )
        .unwrap()
    }

    async fn run_exec(
        join_type: JoinType,
        filter: Option<JoinFilter>,
        relation_type: SpatialRelationType,
    ) -> Vec<RecordBatch> {
        let exec = make_exec(join_type, filter, relation_type);
        let ctx = Arc::new(TaskContext::default());
        let n = exec.right.output_partitioning().partition_count();
        let mut all = Vec::new();
        for p in 0..n {
            let stream = exec.execute(p, Arc::clone(&ctx)).unwrap();
            all.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        }
        all
    }

    #[tokio::test]
    async fn test_join_types_with_filter() {
        let inner = run_exec(
            JoinType::Inner,
            Some(col1_gte_col2_filter()),
            SpatialRelationType::Within,
        )
        .await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 1     | 0     |",
                "| 5     | 2     |",
                "+-------+-------+",
            ],
            &inner
        );

        let left = run_exec(
            JoinType::Left,
            Some(col1_gte_col2_filter()),
            SpatialRelationType::Within,
        )
        .await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 0     |       |",
                "| 1     | 0     |",
                "| 2     |       |",
                "| 3     |       |",
                "| 4     |       |",
                "| 5     | 2     |",
                "| 6     |       |",
                "| 7     |       |",
                "+-------+-------+",
            ],
            &left
        );

        let right = run_exec(
            JoinType::Right,
            Some(col1_gte_col2_filter()),
            SpatialRelationType::Within,
        )
        .await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "|       | 1     |",
                "| 1     | 0     |",
                "|       | 3     |",
                "|       | 4     |",
                "| 5     | 2     |",
                "|       | 5     |",
                "|       | 6     |",
                "|       | 7     |",
                "+-------+-------+",
            ],
            &right
        );
    }

    #[tokio::test]
    async fn test_join_types() {
        let inner = run_exec(JoinType::Inner, None, SpatialRelationType::Within).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 1     | 0     |",
                "| 3     | 5     |",
                "| 3     | 6     |",
                "| 5     | 2     |",
                "+-------+-------+",
            ],
            &inner
        );

        let left = run_exec(JoinType::Left, None, SpatialRelationType::Within).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 0     |       |",
                "| 1     | 0     |",
                "| 2     |       |",
                "| 3     | 5     |",
                "| 3     | 6     |",
                "| 4     |       |",
                "| 5     | 2     |",
                "| 6     |       |",
                "| 7     |       |",
                "+-------+-------+",
            ],
            &left
        );

        let right = run_exec(JoinType::Right, None, SpatialRelationType::Within).await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "|       | 1     |",
                "|       | 3     |",
                "|       | 4     |",
                "|       | 7     |",
                "| 1     | 0     |",
                "| 3     | 5     |",
                "| 3     | 6     |",
                "| 5     | 2     |",
                "+-------+-------+",
            ],
            &right
        );
    }

    #[tokio::test]
    async fn test_dwithin() {
        // distance 0 is effectively st_intersects; every matching build square is
        // strictly inside its probe square, so this collapses to the within rows.
        let zero = run_exec(
            JoinType::Inner,
            None,
            SpatialRelationType::DWithin { distance: 0.0 },
        )
        .await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 1     | 0     |",
                "| 3     | 5     |",
                "| 3     | 6     |",
                "| 5     | 2     |",
                "+-------+-------+",
            ],
            &zero
        );

        // distance 3 additionally matches near-but-disjoint pairs:
        // build[3]=[6,7]^2 to probe[0]=[0,5]^2 (~1.41) -> (3, 0)
        // build[1]=[1,2]^2 to probe[6]=[4,12]^2 (~2.83) -> (1, 6)
        let three = run_exec(
            JoinType::Inner,
            None,
            SpatialRelationType::DWithin { distance: 3.0 },
        )
        .await;
        assert_batches_sorted_eq!(
            [
                "+-------+-------+",
                "| col_1 | col_2 |",
                "+-------+-------+",
                "| 1     | 0     |",
                "| 1     | 6     |",
                "| 3     | 0     |",
                "| 3     | 5     |",
                "| 3     | 6     |",
                "| 5     | 2     |",
                "+-------+-------+",
            ],
            &three
        );
    }
}
