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

use std::cmp::min;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use async_stream::try_stream;
use datafusion::physical_plan::PhysicalExpr;
use futures::stream::{BoxStream, Stream};
#[cfg(test)]
use futures::FutureExt;
use itertools::Itertools;
use zarrs_storage::AsyncReadableListableStorageTraits;

use super::filter::ZarrChunkFilter;
use super::metrics::ZarrMetrics;
use super::util::ColumnNames;
use super::zarr_errors::{ZarrQueryError, ZarrQueryResult};
use super::zarr_store_interface::{
    ChunkPruning, ChunkReadState, SharedZarrStoreInterface, ZarrInMemoryChunk, ZarrStoreInterface,
};
use crate::geospatial::probe_pruning_expr::ProbePruningExpr;

//***************************************************
// A stream of RecordBatches read from a Zarr store.
//
// This struct is separate from `ZarrRecordBatchStream`, so that we
// can avoid manually implementing [`Stream`]. Instead, we use the
// `async-stream` crate to convert an async iterable into a stream.
//***************************************************
struct ZarrRecordBatchStreamInner<T: AsyncReadableListableStorageTraits + ?Sized> {
    read_state: ChunkReadState<T>,
    projected_schema_ref: SchemaRef,
    filter: Option<ZarrChunkFilter>,
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    limit: Option<usize>,

    // the store interface, obtained from a future shared with other streams
    // on the first `next_chunk`. only one of the two is set.
    zarr_store: Option<Arc<ZarrStoreInterface<T>>>,
    shared_zarr_store: Option<SharedZarrStoreInterface<T>>,

    // Resolved chunk-pruning expression, comes from the dynamic filter.
    pruning: Option<ChunkPruning>,

    // the chunks this partition reads, set once the store interface resolves.
    chunk_indices: Option<VecDeque<Vec<u64>>>,

    // the number of partitions and this stream's partition. pushed to the stream
    // because the partiions handle different chunk ranges straight from the
    // zarr storage.
    partition: (usize, usize),
}

// Enumerates every chunk index over the N-D grid (last dimension varies fastest,
// matching the row-major layout of the data) and returns the contiguous slice of
// them that belongs to the given partition.
fn partition_chunk_indices(
    grid_shape: &[u64],
    n_partitions: usize,
    partition: usize,
) -> VecDeque<Vec<u64>> {
    let chunk_indices: Vec<Vec<u64>> = grid_shape
        .iter()
        .map(|&n| 0..n)
        .multi_cartesian_product()
        .collect();
    let chunks_per_partitions = chunk_indices.len().div_ceil(n_partitions);
    let start = chunks_per_partitions * partition;
    let end = min(chunks_per_partitions * (partition + 1), chunk_indices.len());

    // this is to handle cases where more partitions than there are
    // chunks to read were requested.
    if end <= start {
        VecDeque::new()
    } else {
        VecDeque::from(chunk_indices[start..end].to_vec())
    }
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrRecordBatchStreamInner<T> {
    // Create a new ZarrRecordBatchStreamInner.
    //
    // This function is intentionally private, as all users should go through
    // [`ZarrRecordBatchStreamBuilder`] instead.
    fn new(
        shared_zarr_store: SharedZarrStoreInterface<T>,
        projected_schema_ref: SchemaRef,
        n_partitions: usize,
        partition: usize,
    ) -> ZarrQueryResult<Self> {
        if partition >= n_partitions {
            return Err(ZarrQueryError::InvalidCompute(
                "Parition number exceeds number of partition in zarr stream".into(),
            ));
        }

        Ok(Self {
            zarr_store: None,
            shared_zarr_store: Some(shared_zarr_store),
            read_state: ChunkReadState::new(ZarrMetrics::disconnected()),
            projected_schema_ref,
            filter: None,
            dynamic_filters: Vec::new(),
            pruning: None,
            chunk_indices: None,
            partition: (n_partitions, partition),
            limit: None,
        })
    }

    // Resolves the store interface from the shared future, and sets the chunk
    // indices this partition reads on the first call.
    async fn resolve_zarr_store(&mut self) -> ZarrQueryResult<Arc<ZarrStoreInterface<T>>> {
        if let Some(zarr_store) = &self.zarr_store {
            return Ok(zarr_store.clone());
        }

        let shared_zarr_store = self.shared_zarr_store.take().ok_or_else(|| {
            ZarrQueryError::InvalidCompute("No zarr store set on the zarr stream".into())
        })?;
        let zarr_store = shared_zarr_store.await?;
        if zarr_store.schema_ref() != &self.projected_schema_ref {
            return Err(ZarrQueryError::InvalidCompute(
                "Zarr store schema doesn't match the zarr stream schema".into(),
            ));
        }

        let (n_partitions, partition) = self.partition;
        self.chunk_indices = Some(partition_chunk_indices(
            zarr_store.chunk_grid_shape(),
            n_partitions,
            partition,
        ));
        self.zarr_store = Some(zarr_store.clone());
        Ok(zarr_store)
    }

    async fn read(
        &mut self,
        zarr_store: &ZarrStoreInterface<T>,
        cols: &[String],
        chunk_idx: Vec<u64>,
        use_cached_value: bool,
        next_chunk_idx: Option<Vec<u64>>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        zarr_store
            .get_chunk(
                cols,
                chunk_idx,
                use_cached_value,
                next_chunk_idx,
                &mut self.read_state,
            )
            .await
    }

    // returns None if no row in the chunk can pass the filter, otherwise the
    // full chunk columns read to evaluate it, so that they're not read again.
    async fn apply_filter(
        &mut self,
        zarr_store: &ZarrStoreInterface<T>,
        filter: &ZarrChunkFilter,
        chunk_idx: &[u64],
    ) -> ZarrQueryResult<Option<ZarrInMemoryChunk>> {
        zarr_store
            .evaluate_filter(filter, chunk_idx, &mut self.read_state)
            .await
    }

    async fn keep_chunk(
        &mut self,
        zarr_store: &ZarrStoreInterface<T>,
        pruning: &ChunkPruning,
        chunk_idx: &[u64],
    ) -> ZarrQueryResult<bool> {
        zarr_store
            .keep_chunk(pruning, chunk_idx, &mut self.read_state)
            .await
    }

    fn with_dynamic_filters(mut self, dynamic_filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        self.dynamic_filters = dynamic_filters;
        self
    }

    fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    // set before any read, so there are no in flight reads to lose.
    fn with_metrics(mut self, metrics: ZarrMetrics) -> Self {
        self.read_state = ChunkReadState::new(metrics);
        self
    }

    /// Resolves the pushed dynamic filters into a chunk-pruning expression, on
    /// the first `next_chunk` call. By this point the spatial join has finished
    /// its build side, so the shared dynamic filter holds the real
    /// `ProbePruningExpr`.
    fn resolve_pruning(&mut self) -> ZarrQueryResult<()> {
        if self.dynamic_filters.is_empty() {
            return Ok(());
        }

        let mut pruning_expr: Option<Arc<dyn PhysicalExpr>> = None;
        for filter in self.dynamic_filters.drain(..) {
            let Some(inner) = filter.snapshot()? else {
                continue;
            };
            if inner.downcast_ref::<ProbePruningExpr>().is_some() {
                if pruning_expr.is_some() {
                    return Err(ZarrQueryError::InvalidCompute(
                        "more than one chunk-pruning expression pushed to the zarr scan".into(),
                    ));
                }
                pruning_expr = Some(inner);
            }
        }

        if let Some(expr) = pruning_expr {
            self.pruning = Some(ChunkPruning::try_new(expr, &self.projected_schema_ref)?);
        }

        Ok(())
    }

    /// Fetch the next chunk, returning None if there are no more chunks.
    pub(crate) async fn next_chunk(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        // the logic here is not trivial so it wararnts a few explanations.
        // if there is a filter to apply, we read whatever data is needed to
        // evaluate it. we do pre fetch chunks here, when calling get_chunk,
        // and we keep going through the chunk indices until we find a chunk
        // where the filter condition is satisfied.
        //
        // when we do find such a chunk, we move on to the next stage, which
        // is to read the data for the actual query. we do save the data we
        // read to evaluate the filter, because some of it might also be
        // requested in the query. we don't request columns if they are already
        // present in the filter data, and then combine the filter data with
        // the data for the chunk for the main query. if there is a filter,
        // we can't pre fetch the data when reading the data for the main
        // query, because the next time we read some data it would be for the
        // fitler, not for the main query.

        // stop early once this partition has produced its row limit (if any).
        if self.limit == Some(0) {
            return Ok(None);
        }

        // wall-clock timer for the whole batch-producing body.
        let total_start = Instant::now();

        let zarr_store = self.resolve_zarr_store().await?;

        // resolve the pushed dynamic filters into a pruning expression once.
        self.resolve_pruning()?;

        let mut chunk_index = self.pop_chunk_idx();
        let mut filter_chunk: Option<ZarrInMemoryChunk> = None;

        // walk chunks until one passes all present gates (pruning, then static
        // filter).
        if self.pruning.is_some() || self.filter.is_some() {
            let filter = self.filter.take();
            let pruning = self.pruning.take();
            while let Some(idx) = chunk_index.clone() {
                filter_chunk = None;

                if let Some(pruning) = &pruning {
                    if !self.keep_chunk(&zarr_store, pruning, &idx).await? {
                        chunk_index = self.pop_chunk_idx();
                        continue;
                    }
                }

                if let Some(filter) = &filter {
                    filter_chunk = self.apply_filter(&zarr_store, filter, &idx).await?;
                    if filter_chunk.is_none() {
                        chunk_index = self.pop_chunk_idx();
                        continue;
                    }
                }

                break;
            }
            self.filter = filter;
            self.pruning = pruning;
        }

        if let Some(chunk_index) = chunk_index {
            // Only interleave the next chunk when there's no filter (static or
            // dynamic), since a filter may skip it.
            let interleave = self.filter.is_none() && self.pruning.is_none();
            let next_chnk_idx = if interleave {
                self.see_chunk_idx()
            } else {
                None
            };
            // the data arrays read for the filter are not read again.
            let cols: Vec<String> = self
                .projected_schema_ref
                .fields()
                .iter_column_names()
                .filter(|name| filter_chunk.as_ref().is_none_or(|c| c.get(name).is_none()))
                .collect();
            // every projected column can already be in the filter's data.
            let mut chunk = if cols.is_empty() {
                ZarrInMemoryChunk::new()
            } else {
                self.read(&zarr_store, &cols, chunk_index, interleave, next_chnk_idx)
                    .await?
            };
            if let Some(filter_chunk) = filter_chunk {
                chunk.combine(filter_chunk);
            }
            let record_batch = chunk.into_record_batch(&self.projected_schema_ref)?;
            let metrics = self.read_state.metrics();
            metrics.inc_chunks_read();
            metrics.add_rows(record_batch.num_rows());
            metrics.add_total_time(total_start.elapsed());

            if let Some(remaining) = self.limit.as_mut() {
                *remaining = remaining.saturating_sub(record_batch.num_rows());
            }
            Ok(Some(record_batch))
        } else {
            self.read_state
                .metrics()
                .add_total_time(total_start.elapsed());
            Ok(None)
        }
    }

    // Convert this into a `ZarrRecordBatchStream`, using the `async-stream` crate to handle the
    // low-level specifics of stream polling.
    fn into_stream(mut self) -> ZarrRecordBatchStream {
        let stream = Box::pin(try_stream! {
            while let Some(batch) = self.next_chunk().await? {
                yield batch;
            }
        });
        ZarrRecordBatchStream { stream }
    }

    fn pop_chunk_idx(&mut self) -> Option<Vec<u64>> {
        let chunk_idx = self.chunk_indices.as_mut().and_then(|c| c.pop_front());
        if chunk_idx.is_some() {
            self.read_state.metrics().inc_chunks_looked_at();
        }
        chunk_idx
    }

    fn see_chunk_idx(&self) -> Option<Vec<u64>> {
        self.chunk_indices.as_ref().and_then(|c| c.front().cloned())
    }

    // adds a filter to avoid reading whole chunks if no values
    // in the corresponding arrays pass the check. this is not to
    // filter out values within a chunk, we rely on datafusion's
    // default filtering for that. basically this here is to handle
    // filter pushdowns.
    fn with_filter(mut self, filter: ZarrChunkFilter) -> ZarrQueryResult<Self> {
        self.filter = Some(filter);
        Ok(self)
    }
}

// An async stream of record batches read from the Zarr store.
//
// This implementation is modeled to be used with the DataFusion
// [`RecordBatchStream`] trait.
//
// [`RecordBatchStream`]: https://docs.rs/datafusion/latest/datafusion/execution/trait.RecordBatchStream.html
pub(crate) struct ZarrRecordBatchStream {
    stream: BoxStream<'static, Result<RecordBatch, ArrowError>>,
}

// A builder for a [`ZarrRecordBatchStream`]. Everything other than the store,
// schema and partitioning is optional: no filter, no dynamic filters, no limit
// and metrics that aren't reported anywhere.
pub(crate) struct ZarrRecordBatchStreamBuilder<T: AsyncReadableListableStorageTraits + ?Sized> {
    shared_zarr_store: SharedZarrStoreInterface<T>,
    schema_ref: SchemaRef,
    n_partitions: usize,
    partition: usize,
    filter: Option<ZarrChunkFilter>,
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    metrics: ZarrMetrics,
    limit: Option<usize>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrRecordBatchStreamBuilder<T> {
    // the schema must be the projected schema the store interface
    // was created with.
    pub(crate) fn new(
        shared_zarr_store: SharedZarrStoreInterface<T>,
        schema_ref: SchemaRef,
        n_partitions: usize,
        partition: usize,
    ) -> Self {
        Self {
            shared_zarr_store,
            schema_ref,
            n_partitions,
            partition,
            filter: None,
            dynamic_filters: Vec::new(),
            metrics: ZarrMetrics::disconnected(),
            limit: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_from_store(
        store: Arc<T>,
        schema_ref: SchemaRef,
        prefix: Option<String>,
        projection: Option<Vec<usize>>,
        n_partitions: usize,
        partition: usize,
    ) -> Self {
        let projected_schema_ref = match projection {
            Some(proj) => Arc::new(schema_ref.project(&proj).unwrap()),
            None => schema_ref,
        };
        let interface_schema_ref = projected_schema_ref.clone();
        let shared_zarr_store = async move {
            ZarrStoreInterface::new(
                store,
                prefix,
                interface_schema_ref,
                &ZarrMetrics::disconnected(),
            )
            .await
            .map(Arc::new)
            .map_err(Arc::new)
        }
        .boxed()
        .shared();
        Self::new(
            shared_zarr_store,
            projected_schema_ref,
            n_partitions,
            partition,
        )
    }

    pub(crate) fn with_filter(mut self, filter: ZarrChunkFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub(crate) fn with_dynamic_filters(
        mut self,
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        self.dynamic_filters = dynamic_filters;
        self
    }

    pub(crate) fn with_metrics(mut self, metrics: ZarrMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub(crate) fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub(crate) async fn build(self) -> ZarrQueryResult<ZarrRecordBatchStream> {
        let mut inner = ZarrRecordBatchStreamInner::new(
            self.shared_zarr_store,
            self.schema_ref,
            self.n_partitions,
            self.partition,
        )?
        .with_dynamic_filters(self.dynamic_filters)
        .with_metrics(self.metrics);

        if let Some(limit) = self.limit {
            inner = inner.with_limit(limit);
        }
        if let Some(filter) = self.filter {
            inner = inner.with_filter(filter)?;
        }

        Ok(ZarrRecordBatchStream {
            stream: inner.into_stream().stream,
        })
    }
}

impl Stream for ZarrRecordBatchStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

#[cfg(test)]
mod zarr_stream_tests {
    use std::collections::HashMap;

    use arrow::array::AsArray;
    use arrow::compute::concat_batches;
    use datafusion::config::ConfigOptions;
    use datafusion::logical_expr::{Operator, ScalarUDF};
    use datafusion::physical_expr::expressions::{col, lit, Column, DynamicFilterPhysicalExpr};
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion::physical_plan::expressions::binary;
    use futures_util::TryStreamExt;
    use serde_json::json;

    use super::*;
    use crate::geospatial::geos::indexed_build_side::test_helpers::make_indexed_build_side;
    use crate::geospatial::udfs::st_point::StPointUdf;
    use crate::test_utils::{
        extract_col, get_local_zarr_store, get_local_zarr_store_3d, get_local_zarr_store_4d,
        get_local_zarr_store_cf_times, get_local_zarr_store_mix_dims,
        get_local_zarr_store_no_coords, get_local_zarr_store_with_attrs, validate_names_and_types,
        validate_primitive_column,
    };
    // a chunk filter for "lat < 41 AND lon > -118".
    fn lat_lon_filter(schema: &SchemaRef) -> ZarrChunkFilter {
        let lat = binary(col("lat", schema).unwrap(), Operator::Lt, lit(41.0), schema).unwrap();
        let lon = binary(
            col("lon", schema).unwrap(),
            Operator::Gt,
            lit(-118.0),
            schema,
        )
        .unwrap();
        let expr = binary(lat, Operator::And, lon, schema).unwrap();
        ZarrChunkFilter::new(&expr, schema.clone()).unwrap()
    }

    #[tokio::test]
    async fn read_data_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );

        // the top right chunk, 3 x 2
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[2],
            &[35., 35., 36., 36., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[2],
            &[-114.0, -113.0, -114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[2],
            &[6.0, 7.0, 14.0, 15.0, 22.0, 23.0],
        );

        // the bottom right chunk, 2 x 2
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[8],
            &[41.0, 41.0, 42.0, 42.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[8],
            &[-114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[8],
            &[54.0, 55.0, 62.0, 63.0],
        );
    }

    #[tokio::test]
    async fn spatial_pruning_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_spatial_pruning").await;
        let store = wrapper.get_store();

        // Build side: two squares, each landing inside one corner chunk of the
        // 3x3 grid (lat 35-42, lon -120 to -113). Their combined bbox spans the
        // whole store, but only chunks (0,0) and (2,2) intersect a square.
        let squares = [
            "POLYGON((-119.5 35.5, -118.5 35.5, -118.5 36.5, -119.5 36.5, -119.5 35.5))",
            "POLYGON((-114.5 40.5, -113.5 40.5, -113.5 41.5, -114.5 41.5, -114.5 40.5))",
        ];
        let index = make_indexed_build_side(&squares, 1, 1, false, "geometry", "val");

        // Probe geometry: st_point(lon, lat) over the store's coordinate columns.
        let st_point = Arc::new(ScalarUDF::from(StPointUdf::default()));
        let probe_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_point",
            st_point,
            vec![
                Arc::new(Column::new("lon", 2)),
                Arc::new(Column::new("lat", 1)),
            ],
            Arc::new(Field::new("geometry", DataType::Binary, true)),
            Arc::new(ConfigOptions::default()),
        ));

        // Wrap the pruning expression in a dynamic filter, as the spatial join does.
        let pruning =
            Arc::new(ProbePruningExpr::new(probe_expr, index, None)) as Arc<dyn PhysicalExpr>;
        let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![
                Arc::new(Column::new("lon", 2)),
                Arc::new(Column::new("lat", 1)),
            ],
            lit(true),
        ));
        dynamic_filter.update(pruning).unwrap();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .with_dynamic_filters(vec![dynamic_filter as Arc<dyn PhysicalExpr>])
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        assert_eq!(records.len(), 2);
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[1],
            &[54.0, 55.0, 62.0, 63.0],
        );
    }

    #[tokio::test]
    async fn read_data_no_coords_test() {
        let (wrapper, schema) = get_local_zarr_store_no_coords(0.0, "data_no_coords").await;
        let store = wrapper.get_store();

        let stream =
            ZarrRecordBatchStreamBuilder::new_from_store(store, schema.clone(), None, None, 1, 0)
                .build()
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("data_1".to_string(), DataType::Float64),
            ("data_2".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 3);

        let batch = concat_batches(&schema, &records).unwrap();

        validate_primitive_column::<Float64Type, f64>(
            "data_1",
            &batch,
            &[0., 1., 2., 3., 4., 5., 6., 7.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data_2",
            &batch,
            &[100., 101., 102., 103., 104., 105., 106., 107.],
        );
    }

    #[tokio::test]
    async fn read_scaled_data_test() {
        let data1_attrs = json!({ "scale_factor": 2, "add_offset": 1 });
        let data2_attrs = json!({ "scale_factor": 3.0, "add_offset": 4.0 });
        let (wrapper, _) = get_local_zarr_store_with_attrs(
            data1_attrs.as_object().cloned(),
            data2_attrs.as_object().cloned(),
            "scaled_data",
        )
        .await;
        let store = wrapper.get_store();

        // unpacking makes both columns floats, including the int data.
        let schema = Arc::new(Schema::new(vec![
            Field::new("data1", DataType::Float64, true),
            Field::new("data2", DataType::Float64, true),
        ]));
        let stream =
            ZarrRecordBatchStreamBuilder::new_from_store(store, schema.clone(), None, None, 1, 0)
                .build()
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("data1".to_string(), DataType::Float64),
            ("data2".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 3);

        let batch = concat_batches(&schema, &records).unwrap();
        validate_primitive_column::<Float64Type, f64>("data1", &batch, &[3., 5., 7., 9., 11.]);
        validate_primitive_column::<Float64Type, f64>(
            "data2",
            &batch,
            &[34., 64., 94., 124., 154.],
        );
    }

    #[tokio::test]
    async fn read_masked_data_test() {
        // the valid range is in packed units, so it's applied before the
        // scale. scaling first would put every value outside of the range.
        let data1_attrs = json!({ "valid_range": [2, 4], "scale_factor": 10 });
        let data2_attrs = json!({ "_FillValue": 20.0, "missing_value": 40.0 });
        let (wrapper, _) = get_local_zarr_store_with_attrs(
            data1_attrs.as_object().cloned(),
            data2_attrs.as_object().cloned(),
            "masked_data",
        )
        .await;
        let store = wrapper.get_store();

        // the scale makes the int data floats.
        let schema = Arc::new(Schema::new(vec![
            Field::new("data1", DataType::Float64, true),
            Field::new("data2", DataType::Float64, true),
        ]));
        let stream =
            ZarrRecordBatchStreamBuilder::new_from_store(store, schema.clone(), None, None, 1, 0)
                .build()
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 3);

        let batch = concat_batches(&schema, &records).unwrap();
        let values = |name: &str| -> Vec<Option<f64>> {
            batch
                .column_by_name(name)
                .unwrap()
                .as_primitive::<Float64Type>()
                .iter()
                .collect()
        };
        assert_eq!(
            values("data1"),
            vec![None, Some(20.), Some(30.), Some(40.), None]
        );
        assert_eq!(
            values("data2"),
            vec![Some(10.), None, Some(30.), None, Some(50.)]
        );
    }

    #[tokio::test]
    async fn read_cf_times_test() {
        let (wrapper, schema) = get_local_zarr_store_cf_times("cf_times_data").await;
        let store = wrapper.get_store();

        let stream =
            ZarrRecordBatchStreamBuilder::new_from_store(store, schema.clone(), None, None, 1, 0)
                .build()
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            (
                "duration".to_string(),
                DataType::Duration(TimeUnit::Nanosecond),
            ),
            (
                "time1".to_string(),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "time2".to_string(),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 3);

        // jan 1 of 2020 to 2024, in nanoseconds since the epoch.
        let batch = concat_batches(&schema, &records).unwrap();
        let times: Vec<i64> = [
            1_577_836_800,
            1_609_459_200,
            1_640_995_200,
            1_672_531_200,
            1_704_067_200,
        ]
        .iter()
        .map(|t| t * 1_000_000_000)
        .collect();
        validate_primitive_column::<TimestampNanosecondType, i64>("time1", &batch, &times);
        validate_primitive_column::<TimestampNanosecondType, i64>("time2", &batch, &times);

        let durations: Vec<i64> = [10, 100, 1_000, 10_000, 100_000]
            .iter()
            .map(|m| m * 60 * 1_000_000_000)
            .collect();
        validate_primitive_column::<DurationNanosecondType, i64>("duration", &batch, &durations);
    }

    #[tokio::test]
    async fn filter_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_filter").await;
        let store = wrapper.get_store();

        let filter = lat_lon_filter(&schema);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .with_filter(filter)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);

        // this tests for the filter push down, which doesn't completely
        // filter out the results, it only drops chunks of data where
        // not a single "row" passes the filter, so the condition weinto()
        // are checking lines up with the data in the chunks, and is
        // a bit different from the WHERE clause.
        assert_eq!(records.len(), 4);
        for batch in records {
            let lat_values = extract_col::<Float64Type>("lat", &batch);
            let lon_values = extract_col::<Float64Type>("lon", &batch);
            assert!(lat_values
                .iter()
                .zip(lon_values.iter())
                .all(|(lat, lon)| *lat < 41.0 && *lon > -118.0));
        }
    }

    // the store is an 8x8 grid split into 3x3 chunks, so a 3x3 = 9 chunk
    // grid enumerated row-major with per-chunk row counts
    // [9, 9, 6, 9, 9, 6, 6, 6, 4]. the chunk that reaches the limit is
    // emitted whole and the total can overshoot.
    #[tokio::test]
    async fn limit_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_limit").await;
        let store = wrapper.get_store();

        async fn count_rows<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
            store: Arc<T>,
            schema: SchemaRef,
            n_partitions: usize,
            partition: usize,
            limit: usize,
        ) -> usize {
            let stream = ZarrRecordBatchStreamBuilder::new_from_store(
                store,
                schema,
                None,
                None,
                n_partitions,
                partition,
            )
            .with_limit(limit)
            .build()
            .await
            .unwrap();

            let records: Vec<_> = stream.try_collect().await.unwrap();
            records.iter().map(|b| b.num_rows()).sum()
        }

        // single partition, limit 8: the first chunk (9 rows) already reaches
        // the limit, so we stop after it -> 9 rows.
        assert_eq!(count_rows(store.clone(), schema.clone(), 1, 0, 8).await, 9);

        // single partition, limit 10: the first chunk (9) isn't enough, the
        // second chunk (9) crosses the limit -> 18 rows.
        assert_eq!(
            count_rows(store.clone(), schema.clone(), 1, 0, 10).await,
            18
        );

        // two partitions, limit 8 each (applied independently per partition):
        // partition 0 reads chunk [9] and stops -> 9. partition 1's chunks are
        // the edge chunks [6, 6, 6, 4], so [6] isn't enough and [6] crosses the
        // limit -> 12. total = 21.
        let p0 = count_rows(store.clone(), schema.clone(), 2, 0, 8).await;
        let p1 = count_rows(store.clone(), schema.clone(), 2, 1, 8).await;
        assert_eq!(p0 + p1, 21);
    }

    #[tokio::test]
    async fn dimension_tests() {
        // this store will have 2d lat coordinates and 1d lon coordinates.
        // that shoudl effecitvely given the same as 1d and 1d.
        let (wrapper, schema) = get_local_zarr_store_mix_dims(0.0, "lat_lon_mixed_dims_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0],
        );
    }

    #[tokio::test]
    async fn read_3d_data_test() {
        // 6 x 5 x 4 data with dims (lat, lon, height), 2 x 2 x 2 chunks, so a
        // 3 x 3 x 2 = 18 chunk grid. lat/lon are 1D coordinates broadcast up to
        // the full 3D chunk, height is the innermost dim.
        let (wrapper, schema) = get_local_zarr_store_3d(0.0, "lat_lon_height_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("height".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 18);

        // the first chunk, full 2 x 2 x 2.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 35., 36., 36., 36., 36.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[-120., -120., -119., -119., -120., -120., -119., -119.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[0],
            &[100., 200., 100., 200., 100., 200., 100., 200.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[0., 1., 4., 5., 20., 21., 24., 25.],
        );

        // the 5th chunk [0, 2, 0] is the first edge chunk: lon has 5 values
        // with a chunk size of 2, so its last chunk is a partial one, only
        // spanning lon index 4.
        validate_primitive_column::<Float64Type, f64>("lat", &records[4], &[35., 35., 36., 36.]);
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[4],
            &[-116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[4],
            &[100., 200., 100., 200.],
        );
        validate_primitive_column::<Float64Type, f64>("data", &records[4], &[16., 17., 36., 37.]);
    }

    #[tokio::test]
    async fn read_4d_data_test() {
        // 6 x 5 x 4 x 3 data with dims (lat, lon, height, time), 2 x 2 x 2 x 2
        // chunks, so a 3 x 3 x 2 x 2 = 36 chunk grid. lat/lon/height/time are 1D
        // coordinates broadcast up to the full 4D chunk, time is a datetime64[s]
        // column decoded as an arrow Timestamp(Second, None).
        let (wrapper, schema) = get_local_zarr_store_4d(0.0, "lat_lon_height_time_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("height".to_string(), DataType::Float64),
            (
                "time".to_string(),
                DataType::Timestamp(TimeUnit::Second, None),
            ),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 36);

        // the first chunk, full 2 x 2 x 2 x 2, spanning lat {0,1}, lon {0,1},
        // height {0,1}, time {0,1}. same broadcasting logic as the 3D case, just
        // with an extra (innermost) dimension changing the repeat/tile counts.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[
                35., 35., 35., 35., 35., 35., 35., 35., 36., 36., 36., 36., 36., 36., 36., 36.,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120., -120., -120., -120., -119., -119., -119., -119., -120., -120., -120., -120.,
                -119., -119., -119., -119.,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[0],
            &[
                100., 100., 200., 200., 100., 100., 200., 200., 100., 100., 200., 200., 100., 100.,
                200., 200.,
            ],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[0],
            &[
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[
                0., 1., 3., 4., 12., 13., 15., 16., 60., 61., 63., 64., 72., 73., 75., 76.,
            ],
        );

        // the 9th chunk [0, 2, 0, 0] is an edge chunk along the 2nd dimension:
        // lon has 5 values with a chunk size of 2, so its last chunk is partial,
        // only spanning lon index 4. the full chunk shape is 2 x 1 x 2 x 2.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[8],
            &[35., 35., 35., 35., 36., 36., 36., 36.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[8],
            &[-116., -116., -116., -116., -116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[8],
            &[100., 100., 200., 200., 100., 100., 200., 200.],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[8],
            &[
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
                1_700_000_000,
                1_700_000_001,
            ],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[8],
            &[48., 49., 51., 52., 108., 109., 111., 112.],
        );

        // the last chunk [2, 2, 1, 1] is partial along two dimensions at once:
        // lon (size 5) and time (size 3) are both non-multiples of the chunk
        // size 2, so each collapses to width 1. lat and height stay full width.
        // the full chunk shape is 2 x 1 x 2 x 1.
        validate_primitive_column::<Float64Type, f64>("lat", &records[35], &[39., 39., 40., 40.]);
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[35],
            &[-116., -116., -116., -116.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "height",
            &records[35],
            &[300., 400., 300., 400.],
        );
        validate_primitive_column::<TimestampSecondType, i64>(
            "time",
            &records[35],
            &[1_700_000_002, 1_700_000_002, 1_700_000_002, 1_700_000_002],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[35],
            &[296., 299., 356., 359.],
        );
    }

    #[tokio::test]
    async fn read_missing_chunks_test() {
        let fillvalue = 1234.0;
        let (wrapper, schema) = get_local_zarr_store(false, fillvalue, "lat_lon_empty_data").await;
        let store = wrapper.get_store();

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 9);

        // the top left chunk, full 3x3, but "data" is missing.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[35., 35., 35., 36., 36., 36., 37., 37., 37.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[
                -120.0, -119.0, -118.0, -120.0, -119.0, -118.0, -120.0, -119.0, -118.0,
            ],
        );
        validate_primitive_column::<Float64Type, f64>("data", &records[0], &[fillvalue; 9]);
    }

    #[tokio::test]
    async fn read_with_partition_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_with_partition").await;
        let store = wrapper.get_store();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("data".to_string(), DataType::Float64),
        ]);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(
            store.clone(),
            schema.clone(),
            None,
            None,
            2,
            0,
        )
        .build()
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 5);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 2, 1)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        validate_names_and_types(&target_types, &records[0]);
        assert_eq!(records.len(), 4);

        // the full data has 3x3 chunks, the first partition would
        // read the first 5, the second one the last 4, so the first
        // chunk of the second stream would effectively be the middle
        // right chunk of the full data.
        validate_primitive_column::<Float64Type, f64>(
            "lat",
            &records[0],
            &[38., 38., 39., 39., 40., 40.],
        );
        validate_primitive_column::<Float64Type, f64>(
            "lon",
            &records[0],
            &[-114.0, -113.0, -114.0, -113.0, -114.0, -113.0],
        );
        validate_primitive_column::<Float64Type, f64>(
            "data",
            &records[0],
            &[30.0, 31.0, 38.0, 39.0, 46.0, 47.0],
        );
    }

    #[tokio::test]
    async fn read_too_many_partitions_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "lat_lon_data_too_many_partition").await;
        let store = wrapper.get_store();

        // there are only 9 chunks, asking for 20 partitions, so each partition up to
        // the 9th parittion should have one batch in them, after that there should be
        // no data returned by the streams.
        let stream = ZarrRecordBatchStreamBuilder::new_from_store(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            0,
        )
        .build()
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            8,
        )
        .build()
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 1);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(
            store.clone(),
            schema.clone(),
            None,
            None,
            20,
            10,
        )
        .build()
        .await
        .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);

        let stream =
            ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 20, 19)
                .build()
                .await
                .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 0);
    }

    #[tokio::test]
    async fn metrics_test() {
        use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;

        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "lat_lon_data_with_metrics").await;
        let store = wrapper.get_store();

        let filter = lat_lon_filter(&schema);

        // the metrics set is created here so we can read the registered
        // metrics back out after draining the stream.
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = ZarrMetrics::new(&metrics_set, 0);

        let stream = ZarrRecordBatchStreamBuilder::new_from_store(store, schema, None, None, 1, 0)
            .with_filter(filter)
            .with_metrics(metrics)
            .build()
            .await
            .unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(records.len(), 4);

        let snapshot = metrics_set.clone_inner();

        for name in [
            "io_time",
            "decode_time",
            "total_time",
            "chunks_looked_at",
            "chunks_read",
            "rows_produced",
        ] {
            assert!(
                snapshot.sum_by_name(name).is_some(),
                "metric {name} missing from the metrics set"
            );
        }

        // we can't assert anything meaningful on the timings, but the counts are
        // deterministic: all 9 chunks are examined, 4 pass the filter, and those
        // 4 chunks (two of which are 3x2 edge chunks on the right of the grid)
        // hold 30 rows total.
        let value = |name: &str| snapshot.sum_by_name(name).unwrap().as_usize();
        assert_eq!(value("chunks_looked_at"), 9);
        assert_eq!(value("chunks_read"), 4);
        assert_eq!(value("rows_produced"), 30);
    }
}
