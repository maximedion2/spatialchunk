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

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_schema::{FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{GetExt, ScalarValue, Statistics};
use datafusion::config::ConfigOptions;
use datafusion::datasource::file_format::parquet::{ParquetFormat, ParquetFormatFactory};
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::cache::cache_manager::FileMetadataCache;
use datafusion::parquet::basic::LogicalType;
use datafusion::parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use datafusion::parquet::file::statistics::Statistics as ParquetStatistics;
use datafusion::physical_expr::expressions::{Column, DynamicFilterPhysicalExpr, Literal};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::utils::split_conjunction;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, LexRequirement, PhysicalSortExpr, ScalarFunctionExpr,
};
use datafusion::physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{
    DisplayFormatType, ExecutionPlan, PhysicalExpr, SortOrderPushdownResult,
};
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::{FileFormat, FileFormatFactory, FileMeta};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_datasource::morsel::{MorselPlan, MorselPlanner, Morselizer};
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::{PartitionedFile, TableSchema};
use datafusion_datasource_parquet::metadata::DFParquetMetadata;
use datafusion_datasource_parquet::{CachedParquetFileReaderFactory, ParquetAccessPlan};
use delegate::delegate;
use futures::{StreamExt, TryStreamExt};
use geo_types::{coord, Rect};
use object_store::{ObjectMeta, ObjectStore};
use serde_json::Value;

use crate::geospatial::geos::boxed_geo_batch::bbox_of;
use crate::geospatial::join_predicates::SpatialRelationType;
use crate::geospatial::probe_pruning_expr::ProbePruningExpr;
use crate::geospatial::spatial_join_optimizer::literal_f64;
use crate::geospatial::udfs::st_geomfrom::{ARROW_EXTENSION_NAME_KEY, GEOARROW_WKB_EXTENSION_NAME};

// ------------------------------------------------------------------------
// Plans a single geoparquet file. This is one extra step in front of the
// vanilla parquet planner: it fetches the file's footer, builds an access plan
// that skips whatever the spatial filters rule out, attaches it to the file and
// hands the file to the vanilla parquet morselizer.
// ------------------------------------------------------------------------
pub(crate) struct GeoParquetMorselPlanner {
    file: PartitionedFile,

    // the vanilla parquet morselizer, used to create the vanilla planner once
    // the access plan is attached to the file.
    inner: Arc<dyn Morselizer>,

    // to fetch the footer. the metadata cache is shared with the vanilla
    // reader, so the footer is only fetched once per file.
    object_store: Arc<dyn ObjectStore>,
    metadata_size_hint: Option<usize>,
    metadata_cache: Option<Arc<dyn FileMetadataCache>>,

    // the bboxes resolved from the static spatial filters, e.g.
    // st_intersects(geom, <literal>), per geometry column. resolved once at
    // filter pushdown.
    static_bboxes: Arc<StaticBBoxes>,

    // dynamic filters pushed down from parents (e.g. the spatial join's
    // pruning filter). they're placeholders filled in at runtime, the spatial
    // join's build side future updates them once the index is built, and the
    // join only polls the probe side (this) after that future resolves.
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl GeoParquetMorselPlanner {
    pub(crate) fn new(
        file: PartitionedFile,
        inner: Arc<dyn Morselizer>,
        object_store: Arc<dyn ObjectStore>,
        metadata_size_hint: Option<usize>,
        metadata_cache: Option<Arc<dyn FileMetadataCache>>,
        static_bboxes: Arc<StaticBBoxes>,
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            file,
            inner,
            object_store,
            metadata_size_hint,
            metadata_cache,
            static_bboxes,
            dynamic_filters,
        }
    }

    // resolves the dynamic filters into the spatial index pruning expression,
    // if one was pushed. same logic as the zarr reader, the other dynamic
    // filters are ignored.
    fn resolve_pruning(&self) -> Result<Option<Arc<dyn PhysicalExpr>>> {
        let mut pruning_expr: Option<Arc<dyn PhysicalExpr>> = None;
        for filter in &self.dynamic_filters {
            let Some(inner) = filter.snapshot()? else {
                continue;
            };
            if inner.downcast_ref::<ProbePruningExpr>().is_some() {
                if pruning_expr.is_some() {
                    return Err(DataFusionError::Internal(
                        "more than one pruning expression pushed to the geoparquet scan".into(),
                    ));
                }
                pruning_expr = Some(inner);
            }
        }
        Ok(pruning_expr)
    }

    // builds the access plan for the file. if the file bbox is ruled out every
    // row group is skipped, otherwise each row group is checked on its own.
    fn build_access_plan(
        &self,
        metadata: &ParquetMetaData,
        pruning: Option<&ProbePruningExpr>,
    ) -> ParquetAccessPlan {
        let num_row_groups = metadata.num_row_groups();
        if self.file_is_pruned(metadata, pruning) {
            return ParquetAccessPlan::new_none(num_row_groups);
        }

        let mut access_plan = ParquetAccessPlan::new_all(num_row_groups);
        self.prune_row_groups(metadata, pruning, &mut access_plan);
        access_plan
    }

    // the geometry columns that have filters on them, static or dynamic.
    fn filtered_columns<'a>(&'a self, pruning: Option<&'a ProbePruningExpr>) -> Vec<&'a str> {
        let mut columns: Vec<&str> = self.static_bboxes.keys().map(String::as_str).collect();
        if let Some(column) = pruning.and_then(ProbePruningExpr::probe_column) {
            if !columns.contains(&column) {
                columns.push(column);
            }
        }
        columns
    }

    // whether a bbox of a geometry column, for the file or a row group, can
    // contain rows that match the filters on that column.
    fn bbox_may_match(
        &self,
        column: &str,
        bbox: &Rect<f64>,
        pruning: Option<&ProbePruningExpr>,
    ) -> bool {
        if let Some(Some(rect)) = self.static_bboxes.get(column) {
            if !rects_intersect(bbox, rect) {
                return false;
            }
        }
        if let Some(pruning) = pruning {
            if pruning.probe_column() == Some(column) && !pruning.may_intersect(*bbox) {
                return false;
            }
        }
        true
    }

    // whether the whole file can be skipped, either because the static filters
    // can't match at all, or because a file bbox (from the geo metadata) is
    // ruled out by the filters.
    fn file_is_pruned(
        &self,
        metadata: &ParquetMetaData,
        pruning: Option<&ProbePruningExpr>,
    ) -> bool {
        if self.static_bboxes.values().any(Option::is_none) {
            return true;
        }

        self.filtered_columns(pruning).into_iter().any(|column| {
            file_bbox(metadata, column)
                .is_some_and(|bbox| !self.bbox_may_match(column, &bbox, pruning))
        })
    }

    // skips the row groups whose bbox is ruled out by the filters. a row group
    // without a bbox for a column can't be pruned on that column.
    fn prune_row_groups(
        &self,
        metadata: &ParquetMetaData,
        pruning: Option<&ProbePruningExpr>,
        access_plan: &mut ParquetAccessPlan,
    ) {
        // where to read the bboxes from is the same for all row groups.
        let columns: Vec<(&str, Vec<RowGroupBBoxSource>)> = self
            .filtered_columns(pruning)
            .into_iter()
            .map(|column| (column, row_group_bbox_sources(metadata, column)))
            .filter(|(_, sources)| !sources.is_empty())
            .collect();
        if columns.is_empty() {
            return;
        }

        for (idx, row_group) in metadata.row_groups().iter().enumerate() {
            let pruned = columns.iter().any(|(column, sources)| {
                row_group_bbox(row_group, sources)
                    .is_some_and(|bbox| !self.bbox_may_match(column, &bbox, pruning))
            });
            if pruned {
                access_plan.skip(idx);
            }
        }
    }
}

fn rects_intersect(a: &Rect<f64>, b: &Rect<f64>) -> bool {
    // the rect corners, called by path since `min`/`max` are also Ord methods.
    let (a_min, a_max) = (Rect::min(*a), Rect::max(*a));
    let (b_min, b_max) = (Rect::min(*b), Rect::max(*b));
    a_min.x <= b_max.x && b_min.x <= a_max.x && a_min.y <= b_max.y && b_min.y <= a_max.y
}

impl MorselPlanner for GeoParquetMorselPlanner {
    // a single i/o step, fetching the footer, that resolves to the vanilla
    // parquet planner for the file with the access plan attached.
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        Ok(Some(MorselPlan::new().with_pending_planner(async move {
            let metadata =
                DFParquetMetadata::new(self.object_store.as_ref(), &self.file.object_meta)
                    .with_metadata_size_hint(self.metadata_size_hint)
                    .with_file_metadata_cache(self.metadata_cache.clone())
                    .fetch_metadata()
                    .await?;

            let pruning = self.resolve_pruning()?;
            let pruning = pruning
                .as_ref()
                .and_then(|p| p.downcast_ref::<ProbePruningExpr>());
            let access_plan = self.build_access_plan(&metadata, pruning);
            let file = self.file.with_extension(access_plan);
            self.inner.plan_file(file)
        })))
    }
}

impl fmt::Debug for GeoParquetMorselPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeoParquetMorselPlanner")
            .field("file", &self.file.object_meta.location)
            .field("static_bboxes", &self.static_bboxes)
            .field("dynamic_filters", &self.dynamic_filters)
            .finish()
    }
}

// ------------------------------------------------------------------------
// Creates a geoparquet planner for each file. it holds everything the planners
// share, they get the vanilla parquet morselizer to hand the files to.
// ------------------------------------------------------------------------
pub(crate) struct GeoParquetMorselizer {
    inner: Arc<dyn Morselizer>,
    object_store: Arc<dyn ObjectStore>,
    metadata_size_hint: Option<usize>,
    metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    static_bboxes: Arc<StaticBBoxes>,
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl Morselizer for GeoParquetMorselizer {
    fn plan_file(&self, file: PartitionedFile) -> Result<Box<dyn MorselPlanner>> {
        Ok(Box::new(GeoParquetMorselPlanner::new(
            file,
            Arc::clone(&self.inner),
            Arc::clone(&self.object_store),
            self.metadata_size_hint,
            self.metadata_cache.clone(),
            Arc::clone(&self.static_bboxes),
            self.dynamic_filters.clone(),
        )))
    }
}

impl fmt::Debug for GeoParquetMorselizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeoParquetMorselizer")
            .field("inner", &self.inner)
            .field("static_bboxes", &self.static_bboxes)
            .field("dynamic_filters", &self.dynamic_filters)
            .finish()
    }
}

// ------------------------------------------------------------------------
// A file source for (geo)parquet files. it wraps the vanilla parquet source,
// everything is forwarded to it except:
//   - the filter pushdown, where the static spatial filters are resolved into
//     bboxes and the dynamic filters are kept here (and not forwarded).
//   - the morselizer, which is wrapped so that each file goes through the
//     geoparquet planner first, to prune with the bboxes.
// plain parquet files work too, they just have no bboxes to prune with.
// ------------------------------------------------------------------------
#[derive(Clone)]
pub(crate) struct GeoParquetFileSource {
    inner: ParquetSource,
    metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    static_bboxes: Arc<StaticBBoxes>,
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl GeoParquetFileSource {
    pub(crate) fn new(inner: ParquetSource) -> Self {
        Self {
            inner,
            metadata_cache: None,
            static_bboxes: Arc::new(StaticBBoxes::new()),
            dynamic_filters: Vec::new(),
        }
    }

    pub(crate) fn inner(&self) -> &ParquetSource {
        &self.inner
    }

    pub(crate) fn with_inner(mut self, inner: ParquetSource) -> Self {
        self.inner = inner;
        self
    }

    // the cache the footers are fetched through, the same one the vanilla
    // reader uses, so that each footer is fetched once.
    pub(crate) fn with_metadata_cache(
        mut self,
        metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    ) -> Self {
        self.metadata_cache = metadata_cache;
        self
    }

    fn has_spatial_filters(&self) -> bool {
        !self.static_bboxes.is_empty() || !self.dynamic_filters.is_empty()
    }

    // wraps a source returned by the inner one. the inner source is always a
    // parquet source, so this only fails if that changes.
    fn rewrap(&self, source: Arc<dyn FileSource>) -> Result<Arc<dyn FileSource>> {
        let inner = source.downcast_ref::<ParquetSource>().ok_or_else(|| {
            DataFusionError::Internal("expected the inner source to be a parquet source".into())
        })?;
        Ok(Arc::new(self.clone().with_inner(inner.clone())))
    }
}

impl fmt::Debug for GeoParquetFileSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeoParquetFileSource")
            .field("inner", &self.inner)
            .field("static_bboxes", &self.static_bboxes)
            .field("dynamic_filters", &self.dynamic_filters)
            .finish()
    }
}

impl FileSource for GeoParquetFileSource {
    // these are forwarded to the inner parquet source as is.
    delegate! {
        to &self.inner {
            #[through(FileSource)]
            fn create_file_opener(
                &self,
                object_store: Arc<dyn ObjectStore>,
                base_config: &FileScanConfig,
                partition: usize,
            ) -> Result<Arc<dyn FileOpener>>;
            #[through(FileSource)]
            fn table_schema(&self) -> &TableSchema;
            #[through(FileSource)]
            fn filter(&self) -> Option<Arc<dyn PhysicalExpr>>;
            #[through(FileSource)]
            fn projection(&self) -> Option<&ProjectionExprs>;
            #[through(FileSource)]
            fn metrics(&self) -> &ExecutionPlanMetricsSet;
            #[through(FileSource)]
            fn file_type(&self) -> &str;
            #[through(FileSource)]
            fn supports_repartitioning(&self) -> bool;
            #[through(FileSource)]
            fn reorder_files(&self, files: Vec<PartitionedFile>) -> Vec<PartitionedFile>;
        }
    }

    fn create_morselizer(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Box<dyn Morselizer>> {
        let inner =
            self.inner
                .create_morselizer(Arc::clone(&object_store), base_config, partition)?;

        // nothing to prune with, no need to fetch the footers ahead of the
        // vanilla reader.
        if !self.has_spatial_filters() {
            return Ok(inner);
        }

        Ok(Box::new(GeoParquetMorselizer {
            inner: Arc::from(inner),
            object_store,
            metadata_size_hint: self.inner.table_parquet_options().global.metadata_size_hint,
            metadata_cache: self.metadata_cache.clone(),
            static_bboxes: Arc::clone(&self.static_bboxes),
            dynamic_filters: self.dynamic_filters.clone(),
        }))
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        // this can't return an error, and the parquet source always returns a
        // parquet source.
        self.rewrap(self.inner.with_batch_size(batch_size))
            .expect("the parquet source returns a parquet source")
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt_extra(t, f)?;
        if !self.static_bboxes.is_empty() {
            write!(f, ", static_bboxes={:?}", self.static_bboxes)?;
        }
        if !self.dynamic_filters.is_empty() {
            let dynamic: Vec<String> = self.dynamic_filters.iter().map(|d| d.to_string()).collect();
            write!(f, ", dynamic_filters=[{}]", dynamic.join(", "))?;
        }
        Ok(())
    }

    // the static spatial filters are resolved into bboxes and also forwarded,
    // the bbox pruning is inexact so the filters still have to be applied. the
    // dynamic filters are kept here and reported as handled, like the zarr scan.
    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let mut new_source = self.clone();
        let mut forwarded = Vec::new();
        let mut is_dynamic = Vec::with_capacity(filters.len());
        for filter in filters {
            let dynamic = filter.downcast_ref::<DynamicFilterPhysicalExpr>().is_some();
            is_dynamic.push(dynamic);
            if dynamic {
                new_source.dynamic_filters.push(filter);
            } else {
                forwarded.push(filter);
            }
        }

        let mut static_bboxes = (*self.static_bboxes).clone();
        add_static_bboxes(&mut static_bboxes, &forwarded);
        let bboxes_changed = static_bboxes != *self.static_bboxes;
        new_source.static_bboxes = Arc::new(static_bboxes);

        let inner_result = self.inner.try_pushdown_filters(forwarded, config)?;
        let inner_changed = inner_result.updated_node.is_some();
        if let Some(inner) = inner_result.updated_node {
            let inner = inner.downcast_ref::<ParquetSource>().ok_or_else(|| {
                DataFusionError::Internal("expected the inner source to be a parquet source".into())
            })?;
            new_source.inner = inner.clone();
        }

        // the results in the order of the filters, the forwarded ones get
        // whatever the inner source reported.
        let mut inner_results = inner_result.filters.into_iter();
        let results = is_dynamic
            .into_iter()
            .map(|dynamic| {
                if dynamic {
                    PushedDown::Yes
                } else {
                    inner_results.next().unwrap_or(PushedDown::No)
                }
            })
            .collect();

        let propagation = FilterPushdownPropagation::with_parent_pushdown_result(results);
        let dynamic_changed = new_source.dynamic_filters.len() != self.dynamic_filters.len();
        if bboxes_changed || inner_changed || dynamic_changed {
            Ok(propagation.with_updated_node(Arc::new(new_source) as Arc<dyn FileSource>))
        } else {
            Ok(propagation)
        }
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
        eq_properties: &EquivalenceProperties,
    ) -> Result<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        self.inner
            .try_pushdown_sort(order, eq_properties)?
            .try_map(|source| self.rewrap(source))
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        self.inner
            .try_pushdown_projection(projection)?
            .map(|source| self.rewrap(source))
            .transpose()
    }
}
// ------------------------------------------------------------------------
// The (geo)parquet file format. it wraps the vanilla parquet format, and is
// registered in its place (same "parquet" extension), so that STORED AS
// PARQUET goes through it. on top of the vanilla format it:
//   - labels the geometry columns with their geoarrow extension name when
//     inferring the schema, so they're recognized as geometries without
//     having to call st_geomfromwkb.
//   - uses the geoparquet file source, to prune with the bboxes.
// ------------------------------------------------------------------------
#[derive(Debug, Default)]
pub(crate) struct GeoParquetFormat {
    inner: ParquetFormat,
}

impl GeoParquetFormat {
    pub(crate) fn new(inner: ParquetFormat) -> Self {
        Self { inner }
    }
}

// note: delegate! can't be used for the async methods, async_trait rewrites
// them before delegate! is expanded, so those are forwarded by hand.
#[async_trait]
impl FileFormat for GeoParquetFormat {
    delegate! {
        to &self.inner {
            #[through(FileFormat)]
            fn get_ext(&self) -> String;
            #[through(FileFormat)]
            fn get_ext_with_compression(
                &self,
                file_compression_type: &FileCompressionType,
            ) -> Result<String>;
            #[through(FileFormat)]
            fn compression_type(&self) -> Option<FileCompressionType>;
        }
    }

    // the vanilla schema, with the geometry columns labeled. the footers are
    // fetched again to read the geo metadata, they're cached so this doesn't
    // do any i/o.
    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        let schema = self.inner.infer_schema(state, store, objects).await?;

        let metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        let metadata_size_hint = self.inner.metadata_size_hint();
        let metadatas: Vec<Arc<ParquetMetaData>> = futures::stream::iter(objects)
            .map(|object| {
                let metadata_cache = Arc::clone(&metadata_cache);
                async move {
                    DFParquetMetadata::new(store.as_ref(), object)
                        .with_metadata_size_hint(metadata_size_hint)
                        .with_file_metadata_cache(Some(metadata_cache))
                        .fetch_metadata()
                        .await
                }
            })
            .boxed()
            .buffered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect()
            .await?;

        // if files disagree on a column's extension, the first one wins.
        let mut geo_columns: HashMap<String, String> = HashMap::new();
        for metadata in &metadatas {
            for (column, extension) in geo_column_extensions(metadata) {
                geo_columns.entry(column).or_insert(extension);
            }
        }
        if geo_columns.is_empty() {
            return Ok(schema);
        }

        let fields: Vec<FieldRef> = schema
            .fields()
            .iter()
            .map(|field| match geo_columns.get(field.name()) {
                Some(extension) => {
                    let mut field_metadata = field.metadata().clone();
                    field_metadata.insert(ARROW_EXTENSION_NAME_KEY.to_string(), extension.clone());
                    Arc::new(field.as_ref().clone().with_metadata(field_metadata))
                }
                None => Arc::clone(field),
            })
            .collect();
        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        )))
    }

    async fn infer_stats(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        self.inner
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn infer_ordering(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Option<LexOrdering>> {
        self.inner
            .infer_ordering(state, store, table_schema, object)
            .await
    }

    async fn infer_stats_and_ordering(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<FileMeta> {
        self.inner
            .infer_stats_and_ordering(state, store, table_schema, object)
            .await
    }

    // has to be reimplemented because the the vanilla parquet method
    // tries to downcast to a parquet file source, but we want a geo
    // parquet file source here.
    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let source = conf
            .file_source()
            .downcast_ref::<GeoParquetFileSource>()
            .cloned()
            .ok_or_else(|| DataFusionError::Internal("expected a GeoParquetFileSource".into()))?;

        let metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        let store = state
            .runtime_env()
            .object_store(conf.object_store_url.clone())?;

        let mut inner = source
            .inner()
            .clone()
            .with_table_parquet_options(self.inner.options().clone())
            .with_parquet_file_reader_factory(Arc::new(CachedParquetFileReaderFactory::new(
                store,
                Arc::clone(&metadata_cache),
            )));
        if let Some(metadata_size_hint) = self.inner.metadata_size_hint() {
            inner = inner.with_metadata_size_hint(metadata_size_hint);
        }
        let source = source
            .with_inner(inner)
            .with_metadata_cache(Some(metadata_cache));

        let conf = FileScanConfigBuilder::from(conf)
            .with_source(Arc::new(source))
            .build();
        Ok(DataSourceExec::from_data_source(conf))
    }

    // writes plain parquet, without the geo metadata.
    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        state: &dyn Session,
        conf: FileSinkConfig,
        order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner
            .create_writer_physical_plan(input, state, conf, order_requirements)
            .await
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        let inner = ParquetSource::new(table_schema)
            .with_table_parquet_options(self.inner.options().clone());
        Arc::new(GeoParquetFileSource::new(inner))
    }
}

// ------------------------------------------------------------------------
// Creates the geoparquet format. it has the same "parquet" extension as the
// vanilla parquet format factory, to replace it when registered with
// overwrite, so that STORED AS PARQUET goes through the geoparquet format.
// ------------------------------------------------------------------------
#[derive(Debug, Default)]
pub(crate) struct GeoParquetFormatFactory {
    inner: ParquetFormatFactory,
}

impl FileFormatFactory for GeoParquetFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &std::collections::HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let inner = self.inner.create(state, format_options)?;
        let inner = inner.downcast_ref::<ParquetFormat>().ok_or_else(|| {
            DataFusionError::Internal(
                "expected the vanilla factory to create a ParquetFormat".into(),
            )
        })?;
        let inner = ParquetFormat::default().with_options(inner.options().clone());
        Ok(Arc::new(GeoParquetFormat::new(inner)))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(GeoParquetFormat::default())
    }
}

impl GetExt for GeoParquetFormatFactory {
    delegate! {
        to &self.inner {
            #[through(GetExt)]
            fn get_ext(&self) -> String;
        }
    }
}

//***************************************************
// Resolving the static spatial filters into bboxes.
//
// a filter like st_intersects(geom, <literal>) can only match rows whose bbox
// intersects the literal's bbox, so row groups whose bbox doesn't can be
// skipped.
//***************************************************

// Per geometry column, the bbox a row group must intersect for any of its rows
// to match the static filters on that column. None means the filters can't
// match at all so everything can be skipped.
pub(crate) type StaticBBoxes = HashMap<String, Option<Rect<f64>>>;

// Adds the bboxes of the pushed down filters to `bboxes`, per geometry column.
// the conjuncts on the same column are ANDed, so their bboxes are intersected,
// including with the bboxes already there.
pub(crate) fn add_static_bboxes(bboxes: &mut StaticBBoxes, filters: &[Arc<dyn PhysicalExpr>]) {
    for filter in filters {
        for conjunct in split_conjunction(filter) {
            let Some((column, rect)) = spatial_filter_bbox(conjunct) else {
                continue;
            };
            bboxes
                .entry(column)
                .and_modify(|bbox| *bbox = bbox.and_then(|b| rect_intersection(&b, &rect)))
                .or_insert(Some(rect));
        }
    }
}

// The geometry column and bbox of a spatial predicate between a geometry column
// and a literal geometry, in either argument order, e.g.
// st_intersects(geom, <literal>). all the predicates we support imply that the
// bboxes intersect. for st_dwithin, the bbox is expanded by the distance.
//
// the column isn't checked to be a geometry column, the spatial functions
// already check that when the query is planned. and a column that's not a
// geometry column has no bboxes in the footer, so nothing is pruned with it.
fn spatial_filter_bbox(expr: &Arc<dyn PhysicalExpr>) -> Option<(String, Rect<f64>)> {
    let f = expr.downcast_ref::<ScalarFunctionExpr>()?;
    let args = f.args();
    if args.len() < 2 {
        return None;
    }

    let name = f.fun().name();
    let distance = if name.eq_ignore_ascii_case("st_dwithin") {
        literal_f64(args.get(2)?)?
    } else {
        SpatialRelationType::from_name(name)?;
        0.0
    };

    // a closure can't tie its returned references to its arguments, so a fn.
    fn as_pair<'a>(
        a: &'a Arc<dyn PhysicalExpr>,
        b: &'a Arc<dyn PhysicalExpr>,
    ) -> Option<(&'a Column, &'a Literal)> {
        Some((a.downcast_ref::<Column>()?, b.downcast_ref::<Literal>()?))
    }
    let (column, literal) = as_pair(&args[0], &args[1]).or_else(|| as_pair(&args[1], &args[0]))?;

    let rect = literal_bbox(literal.value())?;
    Some((column.name().to_string(), expand_rect(&rect, distance)))
}

// the bbox of a wkb literal. bbox_of computes it in f32, like the spatial
// index, so it can be off by an f32 rounding error.
fn literal_bbox(value: &ScalarValue) -> Option<Rect<f64>> {
    let bytes = match value {
        ScalarValue::Binary(Some(b))
        | ScalarValue::LargeBinary(Some(b))
        | ScalarValue::BinaryView(Some(b)) => b,
        _ => return None,
    };
    let wkb = wkb::reader::read_wkb(bytes).ok()?;
    let rect = bbox_of(&wkb)?;
    Some(Rect::new(
        coord! { x: rect.min().x as f64, y: rect.min().y as f64 },
        coord! { x: rect.max().x as f64, y: rect.max().y as f64 },
    ))
}

fn expand_rect(rect: &Rect<f64>, distance: f64) -> Rect<f64> {
    Rect::new(
        coord! { x: rect.min().x - distance, y: rect.min().y - distance },
        coord! { x: rect.max().x + distance, y: rect.max().y + distance },
    )
}

// None if the rects don't intersect.
fn rect_intersection(a: &Rect<f64>, b: &Rect<f64>) -> Option<Rect<f64>> {
    // the rect corners, called by path since `min`/`max` are also Ord methods.
    let (a_min, a_max) = (Rect::min(*a), Rect::max(*a));
    let (b_min, b_max) = (Rect::min(*b), Rect::max(*b));
    bbox_from_bounds(
        f64::max(a_min.x, b_min.x),
        f64::max(a_min.y, b_min.y),
        f64::min(a_max.x, b_max.x),
        f64::min(a_max.y, b_max.y),
    )
}

//***************************************************
// Reading the bboxes of a geometry column out of the footer, for the file as
// a whole and for each row group. no data is read.
//
// the file bbox comes from the "geo" key-value metadata (geoparquet 1.0+,
// optional). a row group bbox can come from different sources.
//
// anything missing or malformed means no bbox, i.e. no pruning, it never
// fails the read.
//***************************************************

const GEO_METADATA_KEY: &str = "geo";

// Where a row group's bbox can be read from, with the leaf column indices of
// the stats to read. this is the same for every row group in a file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RowGroupBBoxSource {
    // the geospatial statistics of a column with the parquet
    // GEOMETRY/GEOGRAPHY logical type.
    ParquetGeoType {
        col: usize,
    },
    // the min/max of the 4 bbox covering columns.
    Covering {
        xmin: usize,
        ymin: usize,
        xmax: usize,
        ymax: usize,
    },
    // the min/max of the x and y coordinate columns of a geoarrow encoding.
    GeoArrowEncoding {
        x: usize,
        y: usize,
    },
}

// the parsed "geo" key-value metadata, if any.
fn geo_metadata(metadata: &ParquetMetaData) -> Option<Value> {
    let kv = metadata.file_metadata().key_value_metadata()?;
    let geo = kv
        .iter()
        .find(|kv| kv.key == GEO_METADATA_KEY)?
        .value
        .as_ref()?;
    serde_json::from_str(geo).ok()
}

// the column's entry in the "geo" key-value metadata, if any.
fn geo_column_metadata(metadata: &ParquetMetaData, column: &str) -> Option<Value> {
    let mut geo = geo_metadata(metadata)?;
    geo.get_mut("columns")?.get_mut(column).map(Value::take)
}

// The geometry columns of a file, with the geoarrow extension name to label
// them with: geoarrow.wkb for wkb columns, from the geo metadata or with the
// parquet GEOMETRY/GEOGRAPHY logical type, and geoarrow.<encoding> for
// geoarrow encodings, e.g. geoarrow.polygon.
pub(crate) fn geo_column_extensions(metadata: &ParquetMetaData) -> HashMap<String, String> {
    let mut columns = HashMap::new();

    for column in metadata.file_metadata().schema_descr().columns() {
        if let [name] = column.path().parts() {
            if matches!(
                column.logical_type_ref(),
                Some(LogicalType::Geometry { .. } | LogicalType::Geography { .. })
            ) {
                columns.insert(name.clone(), GEOARROW_WKB_EXTENSION_NAME.to_string());
            }
        }
    }

    let geo = geo_metadata(metadata);
    let geo_columns = geo
        .as_ref()
        .and_then(|geo| geo.get("columns"))
        .and_then(Value::as_object);
    for (name, column_meta) in geo_columns.into_iter().flatten() {
        let encoding = column_meta
            .get("encoding")
            .and_then(Value::as_str)
            .unwrap_or("WKB");
        let extension = if encoding.eq_ignore_ascii_case("wkb") {
            GEOARROW_WKB_EXTENSION_NAME.to_string()
        } else {
            format!("geoarrow.{}", encoding.to_lowercase())
        };
        columns.insert(name.clone(), extension);
    }

    columns
}

// the index of the leaf column with exactly this path.
fn leaf_index(metadata: &ParquetMetaData, path: &[String]) -> Option<usize> {
    metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.path().parts() == path)
}

// The file bbox of a geometry column, from the geo metadata. the bbox is either
// [xmin, ymin, xmax, ymax] or, with z, [xmin, ymin, zmin, xmax, ymax, zmax],
// only x/y are kept. a bbox crossing the antimeridian (xmin > xmax) isn't
// handled, `bbox_from_bounds` returns None for it.
pub(crate) fn file_bbox(metadata: &ParquetMetaData, column: &str) -> Option<Rect<f64>> {
    let col_meta = geo_column_metadata(metadata, column)?;
    let bbox: Vec<f64> = col_meta
        .get("bbox")?
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<_>>()?;

    match bbox.as_slice() {
        [xmin, ymin, xmax, ymax] | [xmin, ymin, _, xmax, ymax, _] => {
            bbox_from_bounds(*xmin, *ymin, *xmax, *ymax)
        }
        _ => None,
    }
}

// The sources a geometry column's row group bboxes can be read from, in order
// of preference. empty if there are none.
pub(crate) fn row_group_bbox_sources(
    metadata: &ParquetMetaData,
    column: &str,
) -> Vec<RowGroupBBoxSource> {
    let mut sources = Vec::new();
    let schema = metadata.file_metadata().schema_descr();

    // parquet GEOMETRY/GEOGRAPHY logical type, this doesn't need the geo metadata.
    if let Some(col) = leaf_index(metadata, &[column.to_string()]) {
        if matches!(
            schema.column(col).logical_type_ref(),
            Some(LogicalType::Geometry { .. } | LogicalType::Geography { .. })
        ) {
            sources.push(RowGroupBBoxSource::ParquetGeoType { col });
        }
    }

    let Some(col_meta) = geo_column_metadata(metadata, column) else {
        return sources;
    };

    // the covering, e.g. {"xmin": ["bbox", "xmin"], ...}.
    if let Some(covering) = col_meta.get("covering").and_then(|c| c.get("bbox")) {
        let index = |key: &str| -> Option<usize> {
            let path: Vec<String> = covering
                .get(key)?
                .as_array()?
                .iter()
                .map(|p| p.as_str().map(str::to_string))
                .collect::<Option<_>>()?;
            leaf_index(metadata, &path)
        };
        if let (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) =
            (index("xmin"), index("ymin"), index("xmax"), index("ymax"))
        {
            sources.push(RowGroupBBoxSource::Covering {
                xmin,
                ymin,
                xmax,
                ymax,
            });
        }
    }

    // a geoarrow encoding (anything other than wkb), the coordinates are the
    // leaves under the column named x and y, however deeply nested (the list
    // names vary by writer).
    let encoding = col_meta.get("encoding").and_then(Value::as_str);
    if encoding.is_some_and(|e| !e.eq_ignore_ascii_case("wkb")) {
        let coord_index = |name: &str| {
            schema.columns().iter().position(|c| {
                let parts = c.path().parts();
                parts.len() > 1
                    && parts.first().is_some_and(|p| p == column)
                    && parts.last().is_some_and(|p| p == name)
            })
        };
        if let (Some(x), Some(y)) = (coord_index("x"), coord_index("y")) {
            sources.push(RowGroupBBoxSource::GeoArrowEncoding { x, y });
        }
    }

    sources
}

// The bbox of a row group, from the first source that has one.
pub(crate) fn row_group_bbox(
    row_group: &RowGroupMetaData,
    sources: &[RowGroupBBoxSource],
) -> Option<Rect<f64>> {
    sources.iter().find_map(|source| match source {
        RowGroupBBoxSource::ParquetGeoType { col } => {
            // for geography, xmin > xmax means the bbox wraps around the
            // antimeridian, that's not handled and gives None.
            let bbox = row_group.column(*col).geo_statistics()?.bounding_box()?;
            bbox_from_bounds(
                bbox.get_xmin(),
                bbox.get_ymin(),
                bbox.get_xmax(),
                bbox.get_ymax(),
            )
        }
        RowGroupBBoxSource::Covering {
            xmin,
            ymin,
            xmax,
            ymax,
        } => bbox_from_bounds(
            min_stat(row_group, *xmin)?,
            min_stat(row_group, *ymin)?,
            max_stat(row_group, *xmax)?,
            max_stat(row_group, *ymax)?,
        ),
        RowGroupBBoxSource::GeoArrowEncoding { x, y } => bbox_from_bounds(
            min_stat(row_group, *x)?,
            min_stat(row_group, *y)?,
            max_stat(row_group, *x)?,
            max_stat(row_group, *y)?,
        ),
    })
}

fn bbox_from_bounds(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Option<Rect<f64>> {
    let finite = [xmin, ymin, xmax, ymax].iter().all(|v| v.is_finite());
    if !finite || xmin > xmax || ymin > ymax {
        return None;
    }
    Some(Rect::new(
        coord! { x: xmin, y: ymin },
        coord! { x: xmax, y: ymax },
    ))
}

// the min/max stats of a float or double column, as f64.
fn min_stat(row_group: &RowGroupMetaData, col: usize) -> Option<f64> {
    match row_group.column(col).statistics()? {
        ParquetStatistics::Double(s) => s.min_opt().copied(),
        ParquetStatistics::Float(s) => s.min_opt().map(|v| *v as f64),
        _ => None,
    }
}

fn max_stat(row_group: &RowGroupMetaData, col: usize) -> Option<f64> {
    match row_group.column(col).statistics()? {
        ParquetStatistics::Double(s) => s.max_opt().copied(),
        ParquetStatistics::Float(s) => s.max_opt().map(|v| *v as f64),
        _ => None,
    }
}

#[cfg(test)]
mod geoparquet_tests {
    use std::path::PathBuf;

    use arrow::array::AsArray;
    use arrow::compute::concat_batches;
    use arrow::datatypes::Int32Type;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field};
    use datafusion::execution::object_store::ObjectStoreUrl;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_expr::expressions::lit;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use object_store::path::Path as ObjectPath;
    use object_store::ObjectStoreExt;

    use super::*;
    use crate::geospatial::geos::indexed_build_side::test_helpers::make_indexed_build_side;
    use crate::geospatial::test_utils::{
        wkt_to_wkb, write_geoparquet_covering, write_geoparquet_geoarrow_encoding,
        write_geoparquet_parquet_geo_type, GEOPARQUET_FILE_NAMES,
    };
    use crate::geospatial::StIntersectsUdf;

    // the geometry column, the first one in all the test files.
    fn geometry_column() -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new("geometry", 0))
    }

    // square i of the test data, shifted right and down by half a square, so
    // it intersects squares i and i + 1.
    fn shifted_square_wkt(i: i32) -> String {
        let (x, y) = (i as f64 - 0.5, 11.5 - i as f64);
        format!(
            "POLYGON(({x} {y}, {x1} {y}, {x1} {y1}, {x} {y1}, {x} {y}))",
            x1 = x + 1.0,
            y1 = y + 1.0
        )
    }

    // The static filter and the spatial join pruning expression. The pruning
    // expression starts at first, allowing it to change between tests, sometimes
    // overlapping with the static filter, sometimes not.
    fn spatial_filters(first: i32) -> (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>) {
        let literal = lit(ScalarValue::Binary(Some(wkt_to_wkb(&shifted_square_wkt(
            1,
        )))));
        let static_filter: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_intersects",
            Arc::new(ScalarUDF::from(StIntersectsUdf::default())),
            vec![geometry_column(), literal],
            Arc::new(Field::new("st_intersects", DataType::Boolean, true)),
            Arc::new(ConfigOptions::default()),
        ));

        let build_side = [shifted_square_wkt(first), shifted_square_wkt(first + 1)];
        let build_side: Vec<&str> = build_side.iter().map(String::as_str).collect();
        let index = make_indexed_build_side(&build_side, 1, 1, false, "geometry", "val");
        let pruning: Arc<dyn PhysicalExpr> =
            Arc::new(ProbePruningExpr::new(geometry_column(), index, None));

        (static_filter, pruning)
    }

    // the pruning expression intersects the first row group of the second
    // file, squares 7 to 9.
    fn spatial_filters_second_file() -> (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>) {
        spatial_filters(7)
    }

    // the pruning expression intersects both row groups of the first file,
    // squares 3 to 5. with the static filter, that leaves the first row group.
    fn spatial_filters_first_file() -> (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>) {
        spatial_filters(3)
    }

    // Reads the test files with the geoparquet scan, see `read_parquet_scan`.
    async fn read_with_geoparquet(
        dir_name: &str,
        static_filter: Option<Arc<dyn PhysicalExpr>>,
        pruning: Option<Arc<dyn PhysicalExpr>>,
    ) -> RecordBatch {
        read_parquet_scan(
            dir_name,
            Arc::new(GeoParquetFormat::default()),
            static_filter,
            pruning,
        )
        .await
    }

    // Reads the test files with the vanilla parquet scan, see
    // `read_parquet_scan`. it can't prune with the spatial filters, so it
    // reads everything.
    async fn read_with_vanilla_parquet(
        dir_name: &str,
        static_filter: Option<Arc<dyn PhysicalExpr>>,
        pruning: Option<Arc<dyn PhysicalExpr>>,
    ) -> RecordBatch {
        read_parquet_scan(
            dir_name,
            Arc::new(ParquetFormat::default()),
            static_filter,
            pruning,
        )
        .await
    }

    // Reads the test files in `dir_name` (under the crate's directory) with a
    // parquet scan from `format`, with the given filters pushed down, and
    // returns everything the scan produced in one batch. there's nothing above
    // the scan, so the filters are only used to prune, the batch has every row
    // of the row groups that were read.
    //
    // the pruning expression is wrapped in a dynamic filter, pushed as a
    // placeholder and filled in before the scan runs, like the spatial join
    // does.
    //
    // pushdown_filters is off, so the vanilla scan doesn't evaluate the
    // filters on the decoded batches (the geoparquet pruning doesn't depend
    // on it).
    async fn read_parquet_scan(
        dir_name: &str,
        format: Arc<dyn FileFormat>,
        static_filter: Option<Arc<dyn PhysicalExpr>>,
        pruning: Option<Arc<dyn PhysicalExpr>>,
    ) -> RecordBatch {
        let ctx = SessionContext::new();
        let state = ctx.state();
        let store_url = ObjectStoreUrl::local_filesystem();
        let store = ctx.runtime_env().object_store(&store_url).unwrap();

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir_name);
        let mut objects = Vec::new();
        for name in GEOPARQUET_FILE_NAMES {
            let path = ObjectPath::from_filesystem_path(dir.join(name)).unwrap();
            objects.push(store.head(&path).await.unwrap());
        }

        let schema = format.infer_schema(&state, &store, &objects).await.unwrap();

        let mut filters: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
        if let Some(static_filter) = static_filter {
            filters.push(static_filter);
        }
        let dynamic_filter = pruning.as_ref().map(|_| {
            Arc::new(DynamicFilterPhysicalExpr::new(
                vec![geometry_column()],
                lit(true),
            ))
        });
        if let Some(dynamic_filter) = &dynamic_filter {
            filters.push(Arc::clone(dynamic_filter) as Arc<dyn PhysicalExpr>);
        }

        let mut config = ConfigOptions::default();
        config.execution.parquet.pushdown_filters = false;

        let source = format.file_source(TableSchema::from(Arc::clone(&schema)));
        let pushed = source.try_pushdown_filters(filters, &config).unwrap();
        let source = pushed.updated_node.unwrap_or(source);

        let mut builder = FileScanConfigBuilder::new(store_url, source);
        for object in &objects {
            builder = builder.with_file(PartitionedFile::new(
                object.location.to_string(),
                object.size,
            ));
        }
        let plan = format
            .create_physical_plan(&state, builder.build())
            .await
            .unwrap();

        // the spatial join fills the dynamic filter once its index is built,
        // before the probe side is read.
        if let (Some(dynamic_filter), Some(pruning)) = (dynamic_filter, pruning) {
            dynamic_filter.update(pruning).unwrap();
            dynamic_filter.mark_complete();
        }

        let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
        concat_batches(&plan.schema(), &batches).unwrap()
    }

    // the sorted values of the data column, i.e. the ids of the squares that
    // were read.
    fn data_values(batch: &RecordBatch) -> Vec<i32> {
        let mut values = batch
            .column_by_name("data")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        values.sort();
        values
    }

    // the vanilla reader can't prune with the spatial filters, it reads all
    // the squares.
    #[tokio::test]
    async fn vanilla_reader_test() {
        let dir_name = "geoparquet_vanilla_reader";
        let _dir = write_geoparquet_covering(dir_name);

        let (static_filter, pruning) = spatial_filters_second_file();
        let batch = read_with_vanilla_parquet(dir_name, Some(static_filter), Some(pruning)).await;
        assert_eq!(data_values(&batch), (1..=12).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn covering_pruning_test() {
        let dir_name = "geoparquet_covering_pruning";
        let _dir = write_geoparquet_covering(dir_name);

        // the static filter alone keeps the first row group of the first file.
        let (static_filter, _) = spatial_filters_second_file();
        let batch = read_with_geoparquet(dir_name, Some(static_filter), None).await;
        assert_eq!(data_values(&batch), vec![1, 2, 3]);

        // the dynamic filter alone keeps the first row group of the second file.
        let (_, pruning) = spatial_filters_second_file();
        let batch = read_with_geoparquet(dir_name, None, Some(pruning)).await;
        assert_eq!(data_values(&batch), vec![7, 8, 9]);

        // both together, nothing is left. the second file is pruned with its
        // file bbox, the first file's row groups with their own bboxes.
        let (static_filter, pruning) = spatial_filters_second_file();
        let batch = read_with_geoparquet(dir_name, Some(static_filter), Some(pruning)).await;
        assert_eq!(data_values(&batch), Vec::<i32>::new());

        // the dynamic filter keeps both row groups of the first file, the
        // static filter narrows it down to the first one.
        let (static_filter, pruning) = spatial_filters_first_file();
        let batch = read_with_geoparquet(dir_name, Some(static_filter), Some(pruning)).await;
        assert_eq!(data_values(&batch), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn parquet_geo_type_pruning_test() {
        let dir_name = "geoparquet_parquet_geo_type_pruning";
        let _dir = write_geoparquet_parquet_geo_type(dir_name);

        let (static_filter, pruning) = spatial_filters_first_file();
        let batch = read_with_geoparquet(dir_name, Some(static_filter), Some(pruning)).await;
        assert_eq!(data_values(&batch), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn geoarrow_encoding_pruning_test() {
        let dir_name = "geoparquet_geoarrow_encoding_pruning";
        let _dir = write_geoparquet_geoarrow_encoding(dir_name);

        let (static_filter, pruning) = spatial_filters_first_file();
        let batch = read_with_geoparquet(dir_name, Some(static_filter), Some(pruning)).await;
        assert_eq!(data_values(&batch), vec![1, 2, 3]);
    }
}
