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

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{HashMap, JoinSide, Result, ScalarValue};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::NestedLoopJoinExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, Partitioning, PhysicalExpr,
};

use super::spatial_join_exec::SpatialJoinExec;
use crate::geospatial::join_predicates::{RelationPredicate, SpatialRelationType};
use crate::geospatial::udfs::st_geomfrom::require_geometry;

// A matched spatial predicate plus any leftover (non-spatial) predicate that
// must still be applied as a join filter.
type SpatialPredicateMatch = (RelationPredicate, Option<Arc<dyn PhysicalExpr>>);
#[derive(Debug, Default)]
pub struct SpatialJoinPhysicalOptimizer;

impl SpatialJoinPhysicalOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for SpatialJoinPhysicalOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(plan.transform_up(try_optimize_join)?.data)
    }

    fn name(&self) -> &str {
        "spatial_join_physical_optimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn try_optimize_join(plan: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    // Only nested loops gets reorganized as a spatial join (i.e. not hash
    // joins, those stay the same with the spatial predicate as a filter.)
    let Some(nlj) = plan.downcast_ref::<NestedLoopJoinExec>() else {
        return Ok(Transformed::no(plan));
    };
    let Some(spatial_join) = try_convert_to_spatial_join(nlj)? else {
        return Ok(Transformed::no(plan));
    };
    Ok(Transformed::yes(spatial_join))
}

fn try_convert_to_spatial_join(nlj: &NestedLoopJoinExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(join_filter) = nlj.filter() else {
        return Ok(None);
    };
    let Some((predicate, remainder)) = transform_join_filter(join_filter)? else {
        return Ok(None);
    };

    let left = nlj.left();
    let left = if let Some(coalesce) = left.downcast_ref::<CoalescePartitionsExec>() {
        coalesce.input()
    } else {
        left
    };

    let n_partitions = nlj.right().output_partitioning().partition_count();
    let right = Arc::new(RepartitionExec::try_new(
        nlj.right().clone(),
        Partitioning::RoundRobinBatch(n_partitions),
    )?);

    let exec = SpatialJoinExec::try_new(
        left.clone(),
        right,
        predicate,
        remainder,
        *nlj.join_type(),
        nlj.projection().as_ref().map(|p| p.to_vec()),
    )?;
    Ok(Some(Arc::new(exec)))
}

// Re-arrange the join predicate so that if we have other conditions than
// the spatial join (and the conditions did not give a hash join, at this
// point in the code the original join has to be a nested loop), that other
// condition(s) are moved to a join filter, and the spatial predicate now
// drives the join logic.
fn transform_join_filter(
    jf: &JoinFilter,
) -> Result<Option<(RelationPredicate, Option<JoinFilter>)>> {
    let Some((predicate, remainder_expr)) =
        extract_spatial_predicate(jf.expression(), jf.column_indices(), jf.schema())?
    else {
        return Ok(None);
    };
    let remainder = remainder_expr
        .as_ref()
        .map(|expr| rebuild_join_filter(expr, jf));
    Ok(Some((predicate, remainder)))
}

fn rebuild_join_filter(expr: &Arc<dyn PhysicalExpr>, jf: &JoinFilter) -> JoinFilter {
    let col_refs = collect_column_references(expr, jf.column_indices());
    let referenced: Vec<(usize, &ColumnIndex)> = jf
        .column_indices()
        .iter()
        .enumerate()
        .filter(|(_, ci)| col_refs.contains(ci))
        .collect();
    let pruned_indices: Vec<ColumnIndex> = referenced.iter().map(|(_, ci)| (*ci).clone()).collect();
    let index_remap: HashMap<usize, usize> = referenced
        .iter()
        .enumerate()
        .map(|(new, (old, _))| (*old, new))
        .collect();
    let old_positions: Vec<usize> = referenced.iter().map(|(old, _)| *old).collect();
    let pruned_schema = Arc::new(
        jf.schema()
            .project(&old_positions)
            .expect("failed to project join filter schema"),
    );
    JoinFilter::new(
        reproject_columns(expr, &index_remap),
        pruned_indices,
        pruned_schema,
    )
}

fn extract_spatial_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    col_indices: &[ColumnIndex],
    schema: &Schema,
) -> Result<Option<SpatialPredicateMatch>> {
    if let Some(f) = expr.downcast_ref::<ScalarFunctionExpr>() {
        if let Some(pred) = match_relation_predicate(f, col_indices, schema)? {
            return Ok(Some((pred, None)));
        }
    }

    if let Some(bin) = expr.downcast_ref::<BinaryExpr>() {
        if !matches!(bin.op(), Operator::And) {
            return Ok(None);
        }
        let lhs = bin.left();
        let rhs = bin.right();

        if let Some((pred, rem)) = extract_spatial_predicate(lhs, col_indices, schema)? {
            let remainder = rem.map_or_else(
                || rhs.clone(),
                |r| {
                    Arc::new(BinaryExpr::new(r, Operator::And, rhs.clone()))
                        as Arc<dyn PhysicalExpr>
                },
            );
            return Ok(Some((pred, Some(remainder))));
        }
        if let Some((pred, rem)) = extract_spatial_predicate(rhs, col_indices, schema)? {
            let remainder = rem.map_or_else(
                || lhs.clone(),
                |r| {
                    Arc::new(BinaryExpr::new(lhs.clone(), Operator::And, r))
                        as Arc<dyn PhysicalExpr>
                },
            );
            return Ok(Some((pred, Some(remainder))));
        }
    }

    Ok(None)
}

fn match_relation_predicate(
    f: &ScalarFunctionExpr,
    col_indices: &[ColumnIndex],
    schema: &Schema,
) -> Result<Option<RelationPredicate>> {
    let args = f.args();
    assert!(args.len() >= 2);

    // st_dwithin carries its distance in the third argument, so it isn't derivable
    // from the function name alone. every other predicate maps by name.
    let name = f.fun().name();
    let relation_type = if name.eq_ignore_ascii_case("st_dwithin") {
        match args.get(2).and_then(literal_f64) {
            Some(distance) => SpatialRelationType::DWithin { distance },
            None => return Ok(None),
        }
    } else {
        match SpatialRelationType::from_name(name) {
            Some(rt) => rt,
            None => return Ok(None),
        }
    };

    let refs0 = collect_column_references(&args[0], col_indices);
    let refs1 = collect_column_references(&args[1], col_indices);
    let Some((side0, side1)) = resolve_sides(&refs0, &refs1) else {
        return Ok(None);
    };

    // Now that we know this is a spatial predicate, enforce that both geometry
    // operands are tagged geometry (mirrors the UDF's planning-time check, so a
    // join built without going through the UDFs can't run against bare binary).
    require_geometry(args[0].return_field(schema)?.as_ref())?;
    require_geometry(args[1].return_field(schema)?.as_ref())?;

    let arg0 = reproject_for_side(&args[0], col_indices, side0);
    let arg1 = reproject_for_side(&args[1], col_indices, side1);

    Ok(match (side0, side1) {
        (JoinSide::Left, JoinSide::Right) => {
            Some(RelationPredicate::new(arg0, arg1, relation_type))
        }
        // Arguments reference the join sides in reverse (right-first).  Order by
        // swapping the operands and using the opposite relation.
        (JoinSide::Right, JoinSide::Left) => {
            Some(RelationPredicate::new(arg1, arg0, relation_type.opposite()))
        }
        _ => None,
    })
}

// Extracts a constant f64 from a literal argument. Returns None for a
// non-literal or non-numeric arg.
pub(crate) fn literal_f64(expr: &Arc<dyn PhysicalExpr>) -> Option<f64> {
    let value = expr.downcast_ref::<Literal>()?.value();
    match value.cast_to(&DataType::Float64).ok()? {
        ScalarValue::Float64(Some(v)) => Some(v),
        _ => None,
    }
}

fn collect_column_references(
    expr: &Arc<dyn PhysicalExpr>,
    col_indices: &[ColumnIndex],
) -> Vec<ColumnIndex> {
    let mut out = Vec::new();
    expr.apply(|node| {
        if let Some(col) = node.downcast_ref::<Column>() {
            out.push(col_indices[col.index()].clone());
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .expect("collect_column_references failed");
    out
}

fn resolve_sides(refs0: &[ColumnIndex], refs1: &[ColumnIndex]) -> Option<(JoinSide, JoinSide)> {
    let s0 = uniform_side(refs0)?;
    let s1 = uniform_side(refs1)?;
    if s0 != s1 {
        Some((s0, s1))
    } else {
        None
    }
}

fn uniform_side(refs: &[ColumnIndex]) -> Option<JoinSide> {
    let first = refs.first()?;
    refs.iter()
        .all(|r| r.side == first.side)
        .then_some(first.side)
}

fn reproject_for_side(
    expr: &Arc<dyn PhysicalExpr>,
    col_indices: &[ColumnIndex],
    side: JoinSide,
) -> Arc<dyn PhysicalExpr> {
    if side == JoinSide::None {
        return expr.clone();
    }
    let map: HashMap<usize, usize> = col_indices
        .iter()
        .enumerate()
        .filter_map(|(i, ci)| (ci.side == side).then_some((i, ci.index)))
        .collect();
    reproject_columns(expr, &map)
}

fn reproject_columns(
    expr: &Arc<dyn PhysicalExpr>,
    index_map: &HashMap<usize, usize>,
) -> Arc<dyn PhysicalExpr> {
    expr.clone()
        .transform_down(|node| {
            if let Some(col) = node.downcast_ref::<Column>() {
                if let Some(&new_idx) = index_map.get(&col.index()) {
                    return Ok(Transformed::yes(
                        Arc::new(Column::new(col.name(), new_idx)) as Arc<dyn PhysicalExpr>
                    ));
                }
            }
            Ok(Transformed::no(node))
        })
        .unwrap_or_else(|_| Transformed::no(expr.clone()))
        .data
}

#[cfg(test)]
mod optimizer_tests {
    use std::collections::HashMap;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::JoinType;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_expr::expressions::lit;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::joins::NestedLoopJoinExec;

    use super::super::udfs::{StDWithinUdf, StWithinUdf};
    use super::*;
    use crate::geospatial::join_predicates::SpatialRelationType;
    use crate::geospatial::udfs::st_geomfrom::geometry_field;

    // The field metadata a real geometry column carries (the `geoarrow.wkb`
    // extension tag), reused to build tagged filter-schema fields.
    fn geo_meta() -> HashMap<String, String> {
        geometry_field("g", DataType::Binary, true)
            .metadata()
            .clone()
    }

    fn within_expr(left_idx: usize, right_idx: usize) -> Arc<dyn PhysicalExpr> {
        let udf = Arc::new(ScalarUDF::from(StWithinUdf::default()));
        Arc::new(ScalarFunctionExpr::new(
            "st_within",
            udf,
            vec![
                Arc::new(Column::new("left_geom", left_idx)),
                Arc::new(Column::new("right_geom", right_idx)),
            ],
            Arc::new(Field::new("result", DataType::Boolean, true)),
            Arc::new(ConfigOptions::default()),
        ))
    }

    fn dwithin_expr(left_idx: usize, right_idx: usize, distance: f64) -> Arc<dyn PhysicalExpr> {
        let udf = Arc::new(ScalarUDF::from(StDWithinUdf::default()));
        Arc::new(ScalarFunctionExpr::new(
            "st_dwithin",
            udf,
            vec![
                Arc::new(Column::new("left_geom", left_idx)),
                Arc::new(Column::new("right_geom", right_idx)),
                lit(distance),
            ],
            Arc::new(Field::new("result", DataType::Boolean, true)),
            Arc::new(ConfigOptions::default()),
        ))
    }

    fn geom_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]))
    }

    fn try_run_optimizer(
        join_type: JoinType,
        expr: Arc<dyn PhysicalExpr>,
        metadata: HashMap<String, String>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let left = Arc::new(EmptyExec::new(geom_schema())) as Arc<dyn ExecutionPlan>;
        let right = Arc::new(EmptyExec::new(geom_schema())) as Arc<dyn ExecutionPlan>;
        let geom_field =
            |name: &str| Field::new(name, DataType::Binary, true).with_metadata(metadata.clone());
        let filter_schema = Arc::new(Schema::new(vec![
            geom_field("left_geom"),
            geom_field("right_geom"),
        ]));
        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(expr, column_indices, filter_schema);
        let nlj = NestedLoopJoinExec::try_new(left, right, Some(filter), &join_type, None).unwrap();
        SpatialJoinPhysicalOptimizer::new().optimize(Arc::new(nlj), &ConfigOptions::default())
    }

    fn run_optimizer(join_type: JoinType, expr: Arc<dyn PhysicalExpr>) -> Arc<dyn ExecutionPlan> {
        try_run_optimizer(join_type, expr, geo_meta()).unwrap()
    }

    // The filter schema tags neither geometry column (plain Binary, as if the
    // query skipped st_geomfromwkb/wkt), so the optimizer must reject it.
    #[test]
    fn test_optimizer_rejects_untagged_geometry() {
        let err =
            try_run_optimizer(JoinType::Inner, within_expr(0, 1), HashMap::new()).unwrap_err();
        assert!(
            err.to_string().contains(
                "expected a geometry column tagged 'geoarrow.wkb' \
                 (use st_geomfromwkb / st_geomfromwkt), got a plain Binary column"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_optimizer_converts_nlj_to_spatial_join() {
        let optimized = run_optimizer(JoinType::Inner, within_expr(0, 1));
        let spatial = optimized
            .downcast_ref::<SpatialJoinExec>()
            .expect("should be converted to SpatialJoinExec");
        assert_eq!(spatial.join_type, JoinType::Inner);
        assert_eq!(spatial.predicate.relation_type, SpatialRelationType::Within);
        assert!(spatial.filter.is_none());
    }

    #[test]
    fn test_optimizer_converts_nlj_to_spatial_join_left() {
        let optimized = run_optimizer(JoinType::Left, within_expr(0, 1));
        let spatial = optimized
            .downcast_ref::<SpatialJoinExec>()
            .expect("should be converted to SpatialJoinExec");
        assert_eq!(spatial.join_type, JoinType::Left);
        assert_eq!(spatial.predicate.relation_type, SpatialRelationType::Within);
        assert!(spatial.filter.is_none());
    }

    #[test]
    fn test_optimizer_converts_nlj_to_spatial_join_right() {
        let optimized = run_optimizer(JoinType::Right, within_expr(0, 1));
        let spatial = optimized
            .downcast_ref::<SpatialJoinExec>()
            .expect("should be converted to SpatialJoinExec");
        assert_eq!(spatial.join_type, JoinType::Right);
        assert_eq!(spatial.predicate.relation_type, SpatialRelationType::Within);
        assert!(spatial.filter.is_none());
    }

    #[test]
    fn test_optimizer_converts_dwithin() {
        // The distance literal (third arg) is carried into the relation type.
        let optimized = run_optimizer(JoinType::Inner, dwithin_expr(0, 1, 5.0));
        let spatial = optimized
            .downcast_ref::<SpatialJoinExec>()
            .expect("st_dwithin should convert to a SpatialJoinExec");
        assert_eq!(
            spatial.predicate.relation_type,
            SpatialRelationType::DWithin { distance: 5.0 }
        );
    }

    #[test]
    fn test_optimizer_flips_inverted_args() {
        // st_within(right.geom, left.geom) — args reference the sides in reverse, so the
        // optimizer expresses it left-first as st_contains(left, right) instead of rejecting it.
        let optimized = run_optimizer(JoinType::Inner, within_expr(1, 0));
        let spatial = optimized
            .downcast_ref::<SpatialJoinExec>()
            .expect("inverted st_within(right, left) should convert to a flipped SpatialJoinExec");
        assert_eq!(
            spatial.predicate.relation_type,
            SpatialRelationType::Contains
        );
    }
}
