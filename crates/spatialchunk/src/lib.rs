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

pub mod zarr_store_opener;

pub mod table;

pub mod geospatial;

pub use session::{zarr_session_context, zarr_session_context_with_config};

mod session {
    use std::sync::Arc;

    use datafusion::error::{DataFusionError, Result};
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
    use datafusion::prelude::{SessionConfig, SessionContext};

    use crate::geospatial::file_source::geoparquet_file_source::GeoParquetFormatFactory;
    use crate::geospatial::{
        SpatialJoinPhysicalOptimizer, StAreaUdf, StBufferUdf, StContainsUdf, StCoveredByUdf,
        StCoversUdf, StDWithinUdf, StDistanceUdf, StGeomFromWkbUdf, StGeomFromWktUdf,
        StIntersectsUdf, StPointUdf, StTouchesUdf, StWithinUdf,
    };
    use crate::table::ZarrTableFactory;

    /// A session context set up to query zarr stores and icechunk repos, with
    /// the spatial functions and the spatial join optimization registered.
    pub fn zarr_session_context() -> Result<SessionContext> {
        zarr_session_context_with_config(SessionConfig::new())
    }

    /// Same as [`zarr_session_context`], starting from the given config.
    pub fn zarr_session_context_with_config(config: SessionConfig) -> Result<SessionContext> {
        // the spatial join must replace the nested loop join before the
        // post-phase filter pushdown runs, since that's where the join creates
        // and pushes its dynamic filter.
        let mut rules = PhysicalOptimizer::new().rules;
        let post_pushdown = rules
            .iter()
            .position(|rule| rule.name() == "FilterPushdown(Post)")
            .ok_or_else(|| {
                DataFusionError::Internal("post-phase filter pushdown rule not found".into())
            })?;
        rules.insert(post_pushdown, Arc::new(SpatialJoinPhysicalOptimizer));

        let mut state = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_physical_optimizer_rules(rules)
            .build();
        // replaces the vanilla parquet format (same extension), so that STORED
        // AS PARQUET reads geoparquet. plain parquet files read as before.
        state.register_file_format(Arc::new(GeoParquetFormatFactory::default()), true)?;
        state
            .table_factories_mut()
            .insert("ZARR_STORE".into(), Arc::new(ZarrTableFactory {}));
        #[cfg(feature = "icechunk")]
        state
            .table_factories_mut()
            .insert("ICECHUNK_REPO".into(), Arc::new(ZarrTableFactory {}));

        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(ScalarUDF::from(StAreaUdf::default()));
        ctx.register_udf(ScalarUDF::from(StBufferUdf::default()));
        ctx.register_udf(ScalarUDF::from(StContainsUdf::default()));
        ctx.register_udf(ScalarUDF::from(StCoveredByUdf::default()));
        ctx.register_udf(ScalarUDF::from(StCoversUdf::default()));
        ctx.register_udf(ScalarUDF::from(StDWithinUdf::default()));
        ctx.register_udf(ScalarUDF::from(StDistanceUdf::default()));
        ctx.register_udf(ScalarUDF::from(StGeomFromWkbUdf::default()));
        ctx.register_udf(ScalarUDF::from(StGeomFromWktUdf::default()));
        ctx.register_udf(ScalarUDF::from(StIntersectsUdf::default()));
        ctx.register_udf(ScalarUDF::from(StPointUdf::default()));
        ctx.register_udf(ScalarUDF::from(StTouchesUdf::default()));
        ctx.register_udf(ScalarUDF::from(StWithinUdf::default()));

        Ok(ctx)
    }
}

#[cfg(test)]
mod test_utils {
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::fs;
    use std::num::NonZeroU32;
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow::buffer::ScalarBuffer;
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema, SchemaRef};
    use futures::executor::block_on;
    use itertools::enumerate;
    use ndarray::{Array, Array1, Array2, Array3, Array4};
    use serde_json::{json, Map, Value};
    use walkdir::WalkDir;
    use zarrs::array::data_type::{float64, int64, numpy_datetime64};
    use zarrs::array::{codec, ArrayBuilder, ArraySubset, DataType, Element, FillValue};
    #[cfg(feature = "icechunk")]
    use zarrs::group::GroupBuilder;
    use zarrs::metadata_ext::data_type::NumpyTimeUnit;
    #[cfg(feature = "icechunk")]
    use zarrs_icechunk::icechunk::{ObjectStorage, Repository};
    #[cfg(feature = "icechunk")]
    use zarrs_icechunk::AsyncIcechunkStore;
    use zarrs_object_store::object_store::local::LocalFileSystem;
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::{
        AsyncReadableWritableListableStorageTraits, AsyncWritableStorageTraits, StorePrefix,
    };

    // convenience class to make sure the local zarr stores get cleanup
    // after we're done running a test.
    pub(crate) struct LocalZarrStoreWrapper {
        store: Arc<AsyncObjectStore<LocalFileSystem>>,
        path: PathBuf,
    }

    impl LocalZarrStoreWrapper {
        pub(crate) fn new(store_name: String) -> Self {
            if store_name.is_empty() {
                panic!("name for test zarr store cannot be empty!")
            }

            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
            fs::create_dir(p.clone()).unwrap();
            let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(p.clone()).unwrap());
            Self {
                store: Arc::new(store),
                path: p,
            }
        }

        pub(crate) fn get_store(&self) -> Arc<AsyncObjectStore<LocalFileSystem>> {
            self.store.clone()
        }

        pub(crate) fn get_store_path(&self) -> String {
            self.path.as_os_str().to_str().unwrap().into()
        }
    }

    impl Drop for LocalZarrStoreWrapper {
        fn drop(&mut self) {
            let prefix = StorePrefix::new("").unwrap();
            block_on(self.store.erase_prefix(&prefix)).unwrap();

            while fs::exists(self.path.clone()).unwrap() {
                for d in WalkDir::new(self.path.clone()) {
                    let _ = fs::remove_dir(d.unwrap().path());
                }
            }
        }
    }

    // convenience class to make sure the local icechunk repos get cleanup
    // after we're done running a test.
    #[cfg(feature = "icechunk")]
    pub(crate) struct LocalIcechunkRepoWrapper {
        store: Arc<AsyncIcechunkStore>,
        path: PathBuf,
    }

    #[cfg(feature = "icechunk")]
    impl LocalIcechunkRepoWrapper {
        pub(crate) async fn new(store_name: String) -> Self {
            if store_name.is_empty() {
                panic!("name for test icechunk repo cannot be empty!")
            }
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(store_name);
            fs::create_dir(p.clone()).unwrap();
            let repo = Repository::create(Arc::new(
                ObjectStorage::new_local_filesystem(&p).await.unwrap(),
            ))
            .execute()
            .await
            .unwrap();
            let session = repo.writable_session("main").await.unwrap();
            let store = Arc::new(AsyncIcechunkStore::new(session));
            write_root_group(store.clone()).await;
            Self { store, path: p }
        }

        pub(crate) fn get_store(&self) -> Arc<AsyncIcechunkStore> {
            self.store.clone()
        }

        pub(crate) fn get_store_path(&self) -> String {
            self.path.to_str().unwrap().into()
        }
    }

    #[cfg(feature = "icechunk")]
    impl Drop for LocalIcechunkRepoWrapper {
        fn drop(&mut self) {
            if !self
                .path
                .to_str()
                .unwrap()
                .contains(env!("CARGO_MANIFEST_DIR"))
            {
                panic!("should not be deleting this icechunk repo!")
            }

            // Delete the icechunk repo components one at a time (rather than a
            // bulk remove_dir_all on the whole path) as a safety measure. Not every
            // entry is present in every repo, so tolerate NotFound.
            for dir in [
                "refs",
                "snapshots",
                "manifests",
                "transactions",
                "chunks",
                "overwritten",
            ] {
                match fs::remove_dir_all(self.path.join(dir)) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => panic!("failed to remove {dir}: {e}"),
                }
            }
            for file in ["config.yaml", "repo"] {
                match fs::remove_file(self.path.join(file)) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => panic!("failed to remove {file}: {e}"),
                }
            }
            // Non-recursive: fails loudly if a future icechunk bump adds a new
            // top-level entry, signaling that the list above needs updating.
            fs::remove_dir(self.path.clone()).unwrap();
        }
    }

    // helpers to create some test data on the fly.
    fn get_lz4_compressor() -> codec::BloscCodec {
        codec::BloscCodec::new(
            codec::bytes_to_bytes::blosc::BloscCompressor::LZ4,
            5.try_into().unwrap(),
            Some(0),
            codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
            Some(1),
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn write_1d_array<T>(
        data: Vec<T>,
        data_type: DataType,
        fillvalue: T,
        shape: u64,
        chunk: u64,
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
        attributes: Option<Map<String, Value>>,
    ) where
        T: Element + Clone + Send + Sync,
        FillValue: From<T>,
    {
        let mut array_builder =
            ArrayBuilder::new(vec![shape], [chunk], data_type, FillValue::from(fillvalue));
        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }
        if let Some(attributes) = attributes {
            builder_ref = builder_ref.attributes(attributes);
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        let arr_data: Array1<T> = Array::from_vec(data)
            .into_shape_with_order(shape as usize)
            .unwrap();
        arr.async_store_array_subset(&ArraySubset::new_with_shape(vec![shape]), arr_data)
            .await
            .unwrap();
    }

    pub(crate) async fn write_1d_float_array(
        data: Vec<f64>,
        fillvalue: f64,
        shape: u64,
        chunk: u64,
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        write_1d_array(
            data,
            float64(),
            fillvalue,
            shape,
            chunk,
            store,
            path,
            dimensions,
            None,
        )
        .await;
    }

    pub(crate) async fn write_2d_float_array(
        data: Option<Vec<f64>>,
        fillvalue: f64,
        shape: (u64, u64),
        chunk: (u64, u64),
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape.0, shape.1],
            [chunk.0, chunk.1],
            float64(),
            FillValue::from(fillvalue),
        );

        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        if let Some(data) = data {
            let arr_data: Array2<f64> = Array::from_vec(data)
                .into_shape_with_order((shape.0 as usize, shape.1 as usize))
                .unwrap();
            arr.async_store_array_subset(
                &ArraySubset::new_with_ranges(&[0..shape.0, 0..shape.1]),
                arr_data,
            )
            .await
            .unwrap();
        }
    }

    pub(crate) async fn write_3d_float_array(
        data: Option<Vec<f64>>,
        fillvalue: f64,
        shape: (u64, u64, u64),
        chunk: (u64, u64, u64),
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape.0, shape.1, shape.2],
            [chunk.0, chunk.1, chunk.2],
            float64(),
            FillValue::from(fillvalue),
        );

        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        if let Some(data) = data {
            let arr_data: Array3<f64> = Array::from_vec(data)
                .into_shape_with_order((shape.0 as usize, shape.1 as usize, shape.2 as usize))
                .unwrap();
            arr.async_store_array_subset(
                &ArraySubset::new_with_ranges(&[0..shape.0, 0..shape.1, 0..shape.2]),
                arr_data,
            )
            .await
            .unwrap();
        }
    }

    pub(crate) async fn write_4d_float_array(
        data: Option<Vec<f64>>,
        fillvalue: f64,
        shape: (u64, u64, u64, u64),
        chunk: (u64, u64, u64, u64),
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        path: &str,
        dimensions: Option<Vec<String>>,
    ) {
        let mut array_builder = ArrayBuilder::new(
            vec![shape.0, shape.1, shape.2, shape.3],
            [chunk.0, chunk.1, chunk.2, chunk.3],
            float64(),
            FillValue::from(fillvalue),
        );

        let mut builder_ref = &mut array_builder;
        let codec = get_lz4_compressor();
        builder_ref = builder_ref.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        if let Some(dimensions) = dimensions {
            builder_ref = builder_ref.dimension_names(dimensions.into());
        }

        let arr = builder_ref.build(store, path).unwrap();
        arr.async_store_metadata().await.unwrap();

        if let Some(data) = data {
            let arr_data: Array4<f64> = Array::from_vec(data)
                .into_shape_with_order((
                    shape.0 as usize,
                    shape.1 as usize,
                    shape.2 as usize,
                    shape.3 as usize,
                ))
                .unwrap();
            arr.async_store_array_subset(
                &ArraySubset::new_with_ranges(&[0..shape.0, 0..shape.1, 0..shape.2, 0..shape.3]),
                arr_data,
            )
            .await
            .unwrap();
        }
    }

    // helpers to validate test data.
    pub(crate) fn validate_primitive_column<T, U>(col_name: &str, rec: &RecordBatch, targets: &[U])
    where
        T: ArrowPrimitiveType,
        [U]: AsRef<[<T as arrow_array::ArrowPrimitiveType>::Native]>,
        U: Debug,
    {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(rec.column(idx).as_primitive::<T>().values(), targets);
                matched = true;
            }
        }
        assert!(matched);
    }

    pub(crate) fn validate_names_and_types(
        targets: &HashMap<String, arrow_schema::DataType>,
        rec: &RecordBatch,
    ) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(field.data_type(), targets.get(field.name()).unwrap());
        }
    }

    pub(crate) fn extract_col<T>(col_name: &str, rec_batch: &RecordBatch) -> ScalarBuffer<T::Native>
    where
        T: ArrowPrimitiveType,
    {
        rec_batch
            .column_by_name(col_name)
            .unwrap()
            .as_primitive::<T>()
            .values()
            .clone()
    }

    async fn write_lat_lon_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        write_data: bool,
        fillvalue: f64,
    ) {
        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0];
        write_1d_float_array(
            lats,
            0.0,
            8,
            3,
            store.clone(),
            "/lat",
            Some(["lat".into()].to_vec()),
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1d_float_array(
            lons,
            0.0,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
        )
        .await;

        let data: Option<Vec<_>> = if write_data {
            Some((0..64).map(|i| i as f64).collect())
        } else {
            None
        };
        write_2d_float_array(
            data,
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;
    }

    async fn write_no_coord_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        fillvalue: f64,
    ) {
        let data = (0..8).map(|i| i as f64).collect();
        write_1d_float_array(data, fillvalue, 8, 3, store.clone(), "/data_1", None).await;

        let data = (100..108).map(|i| i as f64).collect();
        write_1d_float_array(data, fillvalue, 8, 3, store.clone(), "/data_2", None).await;
    }

    async fn write_mixed_dims_lat_lon_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        fillvalue: f64,
    ) {
        let lats = [
            vec![35.0; 8],
            vec![36.0; 8],
            vec![37.0; 8],
            vec![38.0; 8],
            vec![39.0; 8],
            vec![40.0; 8],
            vec![41.0; 8],
            vec![42.0; 8],
        ]
        .concat();
        write_2d_float_array(
            Some(lats),
            0.0,
            (8, 8),
            (3, 3),
            store.clone(),
            "/lat",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;

        let lons = vec![
            -120.0, -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0,
        ];
        write_1d_float_array(
            lons,
            0.0,
            8,
            3,
            store.clone(),
            "/lon",
            Some(["lon".into()].to_vec()),
        )
        .await;

        let data = (0..64).map(|i| i as f64).collect();
        write_2d_float_array(
            Some(data),
            fillvalue,
            (8, 8),
            (3, 3),
            store.clone(),
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;
    }

    async fn write_lat_lon_height_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        fillvalue: f64,
    ) {
        // distinct sizes per dimension (6 x 5 x 4) so a bug that mixes up
        // axes can't hide behind equal dimension lengths.
        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0];
        write_1d_float_array(
            lats,
            0.0,
            6,
            2,
            store.clone(),
            "/lat",
            Some(vec!["lat".into()]),
        )
        .await;

        let lons = vec![-120.0, -119.0, -118.0, -117.0, -116.0];
        write_1d_float_array(
            lons,
            0.0,
            5,
            2,
            store.clone(),
            "/lon",
            Some(vec!["lon".into()]),
        )
        .await;

        let heights = vec![100.0, 200.0, 300.0, 400.0];
        write_1d_float_array(
            heights,
            0.0,
            4,
            2,
            store.clone(),
            "/height",
            Some(vec!["height".into()]),
        )
        .await;

        let data = (0..120).map(|i| i as f64).collect();
        write_3d_float_array(
            Some(data),
            fillvalue,
            (6, 5, 4),
            (2, 2, 2),
            store.clone(),
            "/data",
            Some(vec!["lat".into(), "lon".into(), "height".into()]),
        )
        .await;
    }

    async fn write_lat_lon_height_time_data_to_store(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        fillvalue: f64,
    ) {
        // distinct sizes per dimension (6 x 5 x 4 x 3) so a bug that mixes up
        // axes can't hide behind equal dimension lengths.
        let lats = vec![35.0, 36.0, 37.0, 38.0, 39.0, 40.0];
        write_1d_float_array(
            lats,
            0.0,
            6,
            2,
            store.clone(),
            "/lat",
            Some(vec!["lat".into()]),
        )
        .await;

        let lons = vec![-120.0, -119.0, -118.0, -117.0, -116.0];
        write_1d_float_array(
            lons,
            0.0,
            5,
            2,
            store.clone(),
            "/lon",
            Some(vec!["lon".into()]),
        )
        .await;

        let heights = vec![100.0, 200.0, 300.0, 400.0];
        write_1d_float_array(
            heights,
            0.0,
            4,
            2,
            store.clone(),
            "/height",
            Some(vec!["height".into()]),
        )
        .await;

        // time is a numpy.datetime64 with seconds resolution, storing epoch-seconds;
        // the exact values don't matter.
        let times: Vec<i64> = vec![1_700_000_000, 1_700_000_001, 1_700_000_002];
        write_1d_array(
            times,
            numpy_datetime64(NumpyTimeUnit::Second, NonZeroU32::new(1).unwrap()),
            0,
            3,
            2,
            store.clone(),
            "/time",
            Some(vec!["time".into()]),
            None,
        )
        .await;

        let data = (0..360).map(|i| i as f64).collect();
        write_4d_float_array(
            Some(data),
            fillvalue,
            (6, 5, 4, 3),
            (2, 2, 2, 2),
            store.clone(),
            "/data",
            Some(vec![
                "lat".into(),
                "lon".into(),
                "height".into(),
                "time".into(),
            ]),
        )
        .await;
    }

    pub(crate) async fn get_local_zarr_store_3d(
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_lat_lon_height_data_to_store(store, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("height", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    pub(crate) async fn get_local_zarr_store_4d(
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_lat_lon_height_time_data_to_store(store, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("height", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
            Field::new(
                "time",
                ArrowDataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
        ]));

        (wrapper, schema)
    }

    pub(crate) async fn get_local_zarr_store(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_lat_lon_data_to_store(store, write_data, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    pub(crate) async fn get_local_zarr_store_no_coords(
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_no_coord_data_to_store(store, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data_1", ArrowDataType::Float64, true),
            Field::new("data_2", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    // an int64 data1 (1 to 5) and a float64 data2 (10.0 to 50.0), with the
    // given attributes. the schema has the raw types, the attributes are up
    // to the test.
    pub(crate) async fn get_local_zarr_store_with_attrs(
        data1_attrs: Option<Map<String, Value>>,
        data2_attrs: Option<Map<String, Value>>,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        let data1: Vec<i64> = (1..=5).collect();
        write_1d_array(
            data1,
            int64(),
            0,
            5,
            2,
            store.clone(),
            "/data1",
            None,
            data1_attrs,
        )
        .await;

        let data2: Vec<f64> = (1..=5).map(|i| i as f64 * 10.0).collect();
        write_1d_array(
            data2,
            float64(),
            0.0,
            5,
            2,
            store.clone(),
            "/data2",
            None,
            data2_attrs,
        )
        .await;

        let schema = Arc::new(Schema::new(vec![
            Field::new("data1", ArrowDataType::Int64, true),
            Field::new("data2", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    // cf encoded times and durations, time1 as int64 and time2 as float64
    // seconds since the epoch, for jan 1 of 2020 to 2024, and an int64
    // duration in minutes. the schema has the types after the conversions.
    pub(crate) async fn get_local_zarr_store_cf_times(
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        let times: Vec<i64> = vec![
            1_577_836_800,
            1_609_459_200,
            1_640_995_200,
            1_672_531_200,
            1_704_067_200,
        ];
        let attrs = json!({ "units": "seconds since 1970-01-01" });
        write_1d_array(
            times.clone(),
            int64(),
            0,
            5,
            2,
            store.clone(),
            "/time1",
            None,
            attrs.as_object().cloned(),
        )
        .await;

        let attrs = json!({
            "units": "seconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
        });
        write_1d_array(
            times.iter().map(|t| *t as f64).collect(),
            float64(),
            0.0,
            5,
            2,
            store.clone(),
            "/time2",
            None,
            attrs.as_object().cloned(),
        )
        .await;

        let durations: Vec<i64> = vec![10, 100, 1_000, 10_000, 100_000];
        let attrs = json!({ "units": "minutes", "dtype": "timedelta64[ns]" });
        write_1d_array(
            durations,
            int64(),
            0,
            5,
            2,
            store.clone(),
            "/duration",
            None,
            attrs.as_object().cloned(),
        )
        .await;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "duration",
                ArrowDataType::Duration(arrow_schema::TimeUnit::Nanosecond),
                true,
            ),
            Field::new(
                "time1",
                ArrowDataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new(
                "time2",
                ArrowDataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
        ]));

        (wrapper, schema)
    }

    pub(crate) async fn get_local_zarr_store_mix_dims(
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalZarrStoreWrapper, SchemaRef) {
        let wrapper = LocalZarrStoreWrapper::new(dir_name.into());
        let store = wrapper.get_store();

        write_mixed_dims_lat_lon_data_to_store(store, fillvalue).await;
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    #[cfg(feature = "icechunk")]
    async fn write_root_group(store: Arc<AsyncIcechunkStore>) {
        GroupBuilder::new()
            .build(store, "/")
            .unwrap()
            .async_store_metadata()
            .await
            .unwrap();
    }

    #[cfg(feature = "icechunk")]
    pub(crate) async fn get_local_icechunk_repo(
        write_data: bool,
        fillvalue: f64,
        dir_name: &str,
    ) -> (LocalIcechunkRepoWrapper, SchemaRef) {
        let wrapper = LocalIcechunkRepoWrapper::new(dir_name.into()).await;
        let store = wrapper.get_store();

        write_lat_lon_data_to_store(store.clone(), write_data, fillvalue).await;
        let _ = store
            .session()
            .write()
            .await
            .commit("some test data")
            .execute()
            .await
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        (wrapper, schema)
    }

    // writes only the "/data" array (matching the data written by
    // write_lat_lon_data_to_store), with the given shift added to the values.
    #[cfg(feature = "icechunk")]
    async fn write_shifted_data_array(
        store: Arc<dyn AsyncReadableWritableListableStorageTraits>,
        shift: f64,
    ) {
        let data: Vec<f64> = (0..64).map(|i| i as f64 + shift).collect();
        write_2d_float_array(
            Some(data),
            0.0,
            (8, 8),
            (3, 3),
            store,
            "/data",
            Some(["lat".into(), "lon".into()].to_vec()),
        )
        .await;
    }

    // builds a local icechunk repo with three commits: the data is written with a
    // shift of 0, then 1, then 2. we record the snapshot id after the first push,
    // and create a tag.
    #[cfg(feature = "icechunk")]
    pub(crate) async fn get_local_icechunk_repo_multiple_commits(
        dir_name: &str,
    ) -> (LocalIcechunkRepoWrapper, SchemaRef, String, String) {
        if dir_name.is_empty() {
            panic!("name for test icechunk repo cannot be empty!")
        }
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir_name);
        fs::create_dir(p.clone()).unwrap();
        let repo = Repository::create(Arc::new(
            ObjectStorage::new_local_filesystem(&p).await.unwrap(),
        ))
        .execute()
        .await
        .unwrap();

        // first push, with a shift of 0. record the snapshot id.
        let store = Arc::new(AsyncIcechunkStore::new(
            repo.writable_session("main").await.unwrap(),
        ));
        write_root_group(store.clone()).await;
        write_lat_lon_data_to_store(store.clone(), true, 0.0).await;
        let snapshot_id = store
            .session()
            .write()
            .await
            .commit("first commit, shift 0")
            .execute()
            .await
            .unwrap();

        // second push, with a shift of 1. tag the resulting snapshot.
        let store = Arc::new(AsyncIcechunkStore::new(
            repo.writable_session("main").await.unwrap(),
        ));
        write_shifted_data_array(store.clone(), 1.0).await;
        let tagged_snapshot = store
            .session()
            .write()
            .await
            .commit("second commit, shift 1")
            .execute()
            .await
            .unwrap();
        repo.create_tag("test_tag", &tagged_snapshot).await.unwrap();

        // third push, with a shift of 2. this becomes the main branch tip.
        let store = Arc::new(AsyncIcechunkStore::new(
            repo.writable_session("main").await.unwrap(),
        ));
        write_shifted_data_array(store.clone(), 2.0).await;
        let _ = store
            .session()
            .write()
            .await
            .commit("third commit, shift 2")
            .execute()
            .await
            .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("data", ArrowDataType::Float64, true),
            Field::new("lat", ArrowDataType::Float64, true),
            Field::new("lon", ArrowDataType::Float64, true),
        ]));

        let wrapper = LocalIcechunkRepoWrapper { store, path: p };

        (
            wrapper,
            schema,
            String::from(&snapshot_id),
            "test_tag".to_string(),
        )
    }
}
