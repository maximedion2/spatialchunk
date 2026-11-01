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
#[cfg(all(feature = "icechunk", feature = "s3"))]
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{Field, Fields, Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::common::Statistics;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::{DataFusionError, Result as DfResult};
use zarrs::array::Array;
use zarrs::node::{async_get_child_nodes, NodeMetadata, NodePath};
// the icechunk crate is used via zarrs_icechunk's re-export (rather than
// a direct dependency) to keep the two in lockstep.
#[cfg(feature = "icechunk")]
use zarrs_icechunk::icechunk::config::RepositoryConfig;
#[cfg(all(feature = "icechunk", feature = "gcs"))]
use zarrs_icechunk::icechunk::config::{GcsBearerCredential, GcsCredentials, GcsStaticCredentials};
#[cfg(all(feature = "icechunk", feature = "s3"))]
use zarrs_icechunk::icechunk::config::{
    S3ChecksumAlgorithm, S3Credentials, S3Options, S3StaticCredentials,
};
#[cfg(feature = "icechunk")]
use zarrs_icechunk::icechunk::{ObjectStorage, Repository};
#[cfg(feature = "icechunk")]
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_metadata::ArrayMetadata;
// object_store is sourced from zarrs_object_store's re-export here (rather
// than a direct dependency) because these stores are wrapped in
// zarrs_object_store's AsyncObjectStore.
#[cfg(feature = "s3")]
use zarrs_object_store::object_store::aws::AmazonS3Builder;
#[cfg(feature = "gcs")]
use zarrs_object_store::object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::AsyncReadableListableStorageTraits;

use crate::zarr_store_opener::util::NumElements;
use crate::zarr_store_opener::zarr_store_interface::{
    arrow_type, is_1d_coordinate_from_meta, DataTransformer,
};

// Cached, projection-independent ingredients for computing scan statistics.
#[derive(Clone, Debug, Default)]
pub(crate) struct ZarrStatsBase {
    column_dim_names: HashMap<String, Vec<String>>,
    column_dims: HashMap<String, Vec<u64>>,
}

impl ZarrStatsBase {
    // Record one array's contribution. The raw shape is always stored,
    // dimension names are stored only when the array has them.
    fn add_array(&mut self, column: &str, shape: &[u64], dim_names: Option<Vec<String>>) {
        self.column_dims.insert(column.to_string(), shape.to_vec());
        if let Some(names) = dim_names {
            self.column_dim_names.insert(column.to_string(), names);
        }
    }

    // Build DataFusion statistics for a projected schema. only `num_rows`
    // is computed, everything else (byte size, per-column stats)
    // is left `Absent`.
    pub(crate) fn to_datafusion_statistics(
        &self,
        projected_schema: &SchemaRef,
    ) -> DfResult<Statistics> {
        let columns: Vec<&str> = projected_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        let stats = Statistics::new_unknown(projected_schema);
        if columns.is_empty() {
            return Ok(stats);
        }

        // mirrors how the scan produces rows: when data arrays are projected the
        // rows span the full data shape (which all data arrays share), when only
        // 1-D coordinates are projected the rows span just their dimensions.
        let mut data_shape: Option<&Vec<u64>> = None;
        let mut coord_rows: u64 = 1;
        for c in columns {
            let Some(shape) = self.column_dims.get(c) else {
                // every array is recorded in `column_dims` at schema inference,
                // and the projection is always a subset of the inferred schema,
                // so a missing entry means the stats base and schema are out of sync.
                return Err(DataFusionError::Internal(format!(
                    "zarr stats base is missing shape info for projected column '{c}'"
                )));
            };
            let dim_names = self.column_dim_names.get(c).map(Vec::as_slice);

            if is_1d_coordinate_from_meta(c, dim_names, shape.len()) {
                coord_rows *= shape[0];
            } else {
                match data_shape {
                    Some(existing) if existing != shape => {
                        return Err(DataFusionError::Execution(format!(
                            "Shape of array {c} doesn't match the shape of the other data arrays"
                        )));
                    }
                    _ => data_shape = Some(shape),
                }
            }
        }

        let num_rows = match data_shape {
            Some(shape) => shape.num_elements(),
            None => coord_rows,
        };

        Ok(stats.with_num_rows(Precision::Exact(num_rows as usize)))
    }
}

/// A zarr table configuration.
#[derive(Clone, Debug)]
pub struct ZarrTableConfig {
    schema_ref: SchemaRef,
    table_url: ZarrTableUrl,
    projection: Option<Vec<usize>>,
    stats_base: Option<ZarrStatsBase>,
}

impl ZarrTableConfig {
    pub(crate) fn new(table_url: ZarrTableUrl, schema_ref: SchemaRef) -> Self {
        Self {
            schema_ref,
            table_url,
            projection: None,
            stats_base: None,
        }
    }

    pub(crate) fn with_stats_base(mut self, stats_base: ZarrStatsBase) -> Self {
        self.stats_base = Some(stats_base);
        self
    }

    // Statistics for the (projected) scan output. Falls back to all-unknown when
    // no stats base was captured (e.g. configs built directly in tests).
    pub(crate) fn statistics(&self) -> DfResult<Statistics> {
        let projected_schema = self.get_projected_schema_ref();
        match &self.stats_base {
            Some(base) => base.to_datafusion_statistics(&projected_schema),
            None => Ok(Statistics::new_unknown(&projected_schema)),
        }
    }

    pub(crate) async fn get_store_pointer_and_prefix(
        &self,
    ) -> DfResult<(
        Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
        Option<String>,
    )> {
        self.table_url.get_store_pointer_and_prefix().await
    }

    pub(crate) fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.projection = Some(projection);
        self
    }

    pub(crate) fn get_schema_ref(&self) -> SchemaRef {
        self.schema_ref.clone()
    }

    pub(crate) fn get_projected_schema_ref(&self) -> SchemaRef {
        if let Some(projection) = &self.projection {
            let projected_fields: Fields = projection
                .iter()
                .map(|&i| self.schema_ref.field(i).clone())
                .collect();
            Arc::new(Schema::new(projected_fields))
        } else {
            self.schema_ref.clone()
        }
    }
}

// Selects which version of an icechunk repo to read: a branch tip,
// a tag, or an exact snapshot id.
#[cfg(feature = "icechunk")]
#[derive(Clone, Debug)]
pub enum IcechunkVersion {
    BranchTip(String),
    RefTag(String),
    SnapshotId(String),
}

#[cfg(feature = "icechunk")]
impl Default for IcechunkVersion {
    fn default() -> Self {
        Self::BranchTip("main".to_string())
    }
}

#[cfg(feature = "icechunk")]
impl IcechunkVersion {
    // Resolve into an icechunk `VersionInfo`, parsing the snapshot id if needed.
    fn to_version_info(&self) -> DfResult<zarrs_icechunk::icechunk::repository::VersionInfo> {
        use zarrs_icechunk::icechunk::format::SnapshotId;
        use zarrs_icechunk::icechunk::repository::VersionInfo;

        Ok(match self {
            Self::BranchTip(branch) => VersionInfo::BranchTipRef(branch.clone()),
            Self::RefTag(tag) => VersionInfo::TagRef(tag.clone()),
            Self::SnapshotId(id) => {
                VersionInfo::SnapshotId(SnapshotId::try_from(id.as_str()).map_err(|e| {
                    DataFusionError::Execution(format!("invalid icechunk snapshot id '{id}': {e}"))
                })?)
            }
        })
    }
}

// The icechunk-specific configuration shared by every icechunk-backed variant
#[cfg(feature = "icechunk")]
#[derive(Clone, Debug, Default)]
pub struct IcechunkConfig {
    version: IcechunkVersion,
    repo_config: RepositoryConfig,
    // A group inside the repo. unlike plain zarr stores, where a group is
    // part of the url, the url of an icechunk repo only locates the repo,
    // so the group is applied as a key prefix on the opened session.
    group: Option<String>,
}

// The S3 connection config for an icechunk repo. This uses icechunk's own
// `S3Options` / `S3Credentials`.
#[cfg(all(feature = "icechunk", feature = "s3"))]
#[derive(Clone, Debug, Default)]
pub struct S3IcechunkOptions {
    options: S3Options,
    credentials: S3Credentials,
}

// We can create a table based on a directory with a supported zarr
// file/folder structure, or from an icechunk repo.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum ZarrTableUrl {
    ZarrStore {
        url: ListingTableUrl,
    },
    #[cfg(feature = "s3")]
    S3Zarr {
        url: ListingTableUrl,
        builder: AmazonS3Builder,
    },
    #[cfg(feature = "gcs")]
    GcsZarr {
        url: ListingTableUrl,
        builder: GoogleCloudStorageBuilder,
    },
    #[cfg(feature = "icechunk")]
    IcechunkRepo {
        url: ListingTableUrl,
        icechunk: IcechunkConfig,
    },
    #[cfg(all(feature = "icechunk", feature = "s3"))]
    S3Icechunk {
        url: ListingTableUrl,
        icechunk: IcechunkConfig,
        s3: S3IcechunkOptions,
    },
    // Only credentials are configurable for gcs icechunk repos: icechunk's
    // `new_gcs` takes its options as a `HashMap` keyed by object_store 0.14's
    // `GoogleConfigKey`, a type spatialchunk can't name (it builds against
    // object_store 0.13 to stay in sync with zarrs). arbitrary gcs options
    // are unsupported for now.
    #[cfg(all(feature = "icechunk", feature = "gcs"))]
    GcsIcechunk {
        url: ListingTableUrl,
        icechunk: IcechunkConfig,
        credentials: GcsCredentials,
    },
}

// Open a read-only icechunk session on an already-constructed object
// storage, applying the repository config and version selector. Shared
// by every icechunk-backed variant.
#[cfg(feature = "icechunk")]
async fn open_icechunk_session(
    object_storage: ObjectStorage,
    icechunk: &IcechunkConfig,
) -> DfResult<Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>> {
    let repo = Repository::open(Arc::new(object_storage))
        .config(icechunk.repo_config.clone())
        .execute()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let version_info = icechunk.version.to_version_info()?;
    let session = repo
        .readonly_session(&version_info)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(Arc::new(AsyncIcechunkStore::new(session)))
}

impl ZarrTableUrl {
    async fn get_store_pointer_and_prefix(
        &self,
    ) -> DfResult<(
        Arc<dyn AsyncReadableListableStorageTraits + Unpin + Send>,
        Option<String>,
    )> {
        // the Option<String> that is returned here requires some explanation.
        // for some remote stores, the full url is not used as a prefix when
        // writing and reading from the store. for example for aws s3, it
        // seems the bucket is extracted from the url, but other than that
        // the object path is not kept, so when reading from the store, you
        // always need to provide a prefix. but for local object stores, it
        // actually can store the prefix. so we need to sometimes return no
        // prefix (None) and sometimes return one (Some(prefix)).
        match self {
            // this is for the case of a directory with a zarr file structure inside.
            Self::ZarrStore { url } => match url.scheme() {
                "file" => {
                    let path = PathBuf::from("/".to_owned() + url.prefix().as_ref());
                    let store = AsyncObjectStore::new(LocalFileSystem::new_with_prefix(path)?);
                    Ok((Arc::new(store), None))
                }
                _ => Err(DataFusionError::Execution(format!(
                    "Unsupported table url scheme {} for zarr stores",
                    url.scheme()
                ))),
            },

            // a zarr store on s3, with a user-configured object-store builder.
            #[cfg(feature = "s3")]
            Self::S3Zarr { url, builder } => {
                let store = builder.clone().with_url(url.get_url().as_str()).build()?;
                let store = AsyncObjectStore::new(store);
                Ok((Arc::new(store), Some(url.prefix().to_string())))
            }

            // a zarr store on gcs, with a user-configured object-store builder.
            #[cfg(feature = "gcs")]
            Self::GcsZarr { url, builder } => {
                let store = builder.clone().with_url(url.get_url().as_str()).build()?;
                let store = AsyncObjectStore::new(store);
                Ok((Arc::new(store), Some(url.prefix().to_string())))
            }

            // an icechunk repo on the local filesystem.
            #[cfg(feature = "icechunk")]
            Self::IcechunkRepo { url, icechunk } => {
                let object_storage = match url.scheme() {
                    "file" => {
                        let path = PathBuf::from("/".to_owned() + url.prefix().as_ref());
                        ObjectStorage::new_local_filesystem(&path)
                            .await
                            .map_err(|e| DataFusionError::External(Box::new(e)))?
                    }
                    _ => {
                        return Err(DataFusionError::Execution(format!(
                            "Unsupported table url scheme {} for icechunk repos",
                            url.scheme()
                        )))
                    }
                };
                Ok((
                    open_icechunk_session(object_storage, icechunk).await?,
                    icechunk.group.clone(),
                ))
            }

            // an icechunk repo on s3, with user-configured icechunk `S3Options` /
            // `S3Credentials`.
            #[cfg(all(feature = "icechunk", feature = "s3"))]
            Self::S3Icechunk { url, icechunk, s3 } => {
                let bucket = url
                    .object_store()
                    .as_str()
                    .replace("s3://", "")
                    .trim_end_matches("/")
                    .to_string();

                // start from the user-provided options, falling back to the
                // AWS_DEFAULT_REGION env var when no region was set explicitly.
                let mut options = s3.options.clone();
                if options.region.is_none() {
                    if let Ok(region) = env::var("AWS_DEFAULT_REGION") {
                        options = options.with_region(region);
                    }
                }

                let object_storage = ObjectStorage::new_s3(
                    bucket,
                    Some(url.prefix().as_ref().to_string()),
                    Some(s3.credentials.clone()),
                    Some(options),
                    Vec::new(),
                    Vec::new(),
                )
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

                Ok((
                    open_icechunk_session(object_storage, icechunk).await?,
                    icechunk.group.clone(),
                ))
            }

            // an icechunk repo on gcs, with user-configured icechunk `GcsCredentials`
            #[cfg(all(feature = "icechunk", feature = "gcs"))]
            Self::GcsIcechunk {
                url,
                icechunk,
                credentials,
            } => {
                let bucket = url
                    .object_store()
                    .as_str()
                    .replace("gs://", "")
                    .trim_end_matches("/")
                    .to_string();

                let object_storage = ObjectStorage::new_gcs(
                    bucket,
                    Some(url.prefix().as_ref().to_string()),
                    Some(credentials.clone()),
                    None,
                    Vec::new(),
                    Vec::new(),
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

                Ok((
                    open_icechunk_session(object_storage, icechunk).await?,
                    icechunk.group.clone(),
                ))
            }
        }
    }

    pub(crate) async fn infer_schema(&self) -> DfResult<(SchemaRef, ZarrStatsBase)> {
        let (store, store_prefix) = self.get_store_pointer_and_prefix().await?;
        let store_prefix = store_prefix
            .as_ref()
            .map_or("".into(), |p| p.to_owned() + "/");

        // a table is made of the arrays directly inside the group it points at,
        // child groups are skipped. only the direct children are listed, a
        // recursive listing would also enumerate every chunk key, which is very
        // expensive on large stores (for icechunk it requires loading every
        // manifest).
        let group_path = NodePath::new(&format!("/{}", store_prefix.trim_end_matches('/')))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let children = async_get_child_nodes(&store, &group_path, false)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // zarr has no inherent column order, so we need to impose a deterministic
        // one. downstream code (and the test helpers) rely on this alphabetical
        // ordering.
        let mut arrays: Vec<(String, ArrayMetadata)> = children
            .iter()
            .filter_map(|node| match node.metadata() {
                NodeMetadata::Array(meta) => Some((node.path().as_str().to_string(), meta.clone())),
                NodeMetadata::Group(_) => None,
            })
            .collect();
        arrays.sort_by(|a, b| a.0.cmp(&b.0));

        let mut fields = Vec::with_capacity(arrays.len());
        let mut stats_base = ZarrStatsBase::default();

        for (array_path, array_meta) in arrays {
            let field_name = array_path
                .rsplit('/')
                .next()
                .unwrap_or(&array_path)
                .to_string();

            let arr = Array::new_with_metadata(store.clone(), &array_path, array_meta)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            if !matches!(arr.metadata(), ArrayMetadata::V3(_)) {
                return Err(DataFusionError::Execution(
                    "Only Zarr v3 metadata is supported".into(),
                ));
            }

            // capture the array's shape and dimension names for statistics.
            let dim_names: Option<Vec<String>> = arr
                .dimension_names()
                .clone()
                .and_then(|names| names.into_iter().collect::<Option<Vec<String>>>());
            stats_base.add_array(&field_name, arr.shape(), dim_names);

            // the schema has the types after the cf attributes transformations,
            // the raw types are only known by the arrays.
            let raw_type =
                arrow_type(arr.data_type()).map_err(|e| DataFusionError::External(Box::new(e)))?;
            let raw_field = Field::new(field_name, raw_type, true);
            let transformer = DataTransformer::new(&raw_field, arr.attributes())
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            fields.push(transformer.output_field(&raw_field));
        }

        Ok((Arc::new(Schema::new(Fields::from(fields))), stats_base))
    }
}

// Parse a user-supplied boolean option value, erroring on anything that isn't
// `true` / `false`.
#[cfg(any(feature = "s3", feature = "gcs"))]
fn parse_bool_option(option: &str, value: &str) -> DfResult<bool> {
    value.parse::<bool>().map_err(|_| {
        DataFusionError::Execution(format!(
            "invalid boolean value '{value}' for option '{option}' (expected true or false)"
        ))
    })
}

// Builder for a plain-zarr `ZarrTableUrl`.
//
// It collects an optional `object_store` builder for whichever cloud
// backend the user configured, from the `s3.*` / `gcs.*` option namespaces.
// s3 and gcs options are mutually exclusive (setting one while the other
// is present errors). When no options were supplied the builder is seeded
// from the environment (`from_env`)
pub struct ZarrUrlBuilder {
    url: ListingTableUrl,
    #[cfg(feature = "s3")]
    s3: Option<AmazonS3Builder>,
    #[cfg(feature = "gcs")]
    gcs: Option<GoogleCloudStorageBuilder>,
}

impl ZarrUrlBuilder {
    // Create a builder for `url`, optionally applying a map of namespaced options
    // (e.g. from `CREATE EXTERNAL TABLE ... OPTIONS(...)`). `None` leaves every
    // backend at its environment default.
    pub fn try_new(
        url: ListingTableUrl,
        options: Option<&HashMap<String, String>>,
    ) -> DfResult<Self> {
        let mut builder = Self {
            url,
            #[cfg(feature = "s3")]
            s3: None,
            #[cfg(feature = "gcs")]
            gcs: None,
        };
        if let Some(options) = options {
            for (key, value) in options {
                builder = builder.apply_option(key, value)?;
            }
        }
        Ok(builder)
    }

    // Route one namespaced option (`<namespace>.<option>`) to the matching setter.
    #[cfg_attr(not(any(feature = "s3", feature = "gcs")), allow(unused_variables))]
    fn apply_option(self, key: &str, value: &str) -> DfResult<Self> {
        let (namespace, option) = key.split_once('.').ok_or_else(|| {
            DataFusionError::Execution(format!(
                "invalid option key '{key}': expected a namespaced key like 's3.region'"
            ))
        })?;
        match namespace {
            #[cfg(feature = "s3")]
            "s3" => self.apply_s3_option(option, value),
            #[cfg(feature = "gcs")]
            "gcs" => self.apply_gcs_option(option, value),
            other => Err(DataFusionError::Execution(format!(
                "unknown or unsupported option namespace '{other}' for a zarr store"
            ))),
        }
    }

    #[cfg(feature = "s3")]
    fn apply_s3_option(self, option: &str, value: &str) -> DfResult<Self> {
        match option {
            "region" => self.with_s3_region(value),
            "endpoint" => self.with_s3_endpoint(value),
            "allow_http" => self.with_s3_allow_http(parse_bool_option(option, value)?),
            "force_path_style" => self.with_s3_force_path_style(parse_bool_option(option, value)?),
            "anonymous" => self.with_s3_anonymous(parse_bool_option(option, value)?),
            "access_key_id" => self.with_s3_access_key_id(value),
            "secret_access_key" => self.with_s3_secret_access_key(value),
            "session_token" => self.with_s3_session_token(value),
            other => Err(DataFusionError::Execution(format!(
                "unknown s3 option '{other}' for a zarr store"
            ))),
        }
    }

    #[cfg(feature = "gcs")]
    fn apply_gcs_option(self, option: &str, value: &str) -> DfResult<Self> {
        match option {
            "service_account" => self.with_gcs_service_account(value),
            "service_account_key" => self.with_gcs_service_account_key(value),
            "application_credentials" => self.with_gcs_application_credentials(value),
            "endpoint" => self.with_gcs_endpoint(value),
            "anonymous" => self.with_gcs_anonymous(parse_bool_option(option, value)?),
            other => Err(DataFusionError::Execution(format!(
                "unknown gcs option '{other}' for a zarr store"
            ))),
        }
    }

    #[cfg(feature = "s3")]
    fn take_s3(&mut self) -> DfResult<AmazonS3Builder> {
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "cannot set both s3 and gcs options on the same zarr store".into(),
            ));
        }
        Ok(self.s3.take().unwrap_or_else(AmazonS3Builder::from_env))
    }

    #[cfg(feature = "gcs")]
    fn take_gcs(&mut self) -> DfResult<GoogleCloudStorageBuilder> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "cannot set both s3 and gcs options on the same zarr store".into(),
            ));
        }
        Ok(self
            .gcs
            .take()
            .unwrap_or_else(GoogleCloudStorageBuilder::from_env))
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_region(mut self, region: impl Into<String>) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_region(region));
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_endpoint(mut self, endpoint: impl Into<String>) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_endpoint(endpoint));
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_allow_http(mut self, allow_http: bool) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_allow_http(allow_http));
        Ok(self)
    }

    // `force_path_style` is the inverse of object_store's virtual-hosted-style flag.
    #[cfg(feature = "s3")]
    pub fn with_s3_force_path_style(mut self, force_path_style: bool) -> DfResult<Self> {
        self.s3 = Some(
            self.take_s3()?
                .with_virtual_hosted_style_request(!force_path_style),
        );
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_anonymous(mut self, anonymous: bool) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_skip_signature(anonymous));
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_access_key_id(mut self, access_key_id: impl Into<String>) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_access_key_id(access_key_id));
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_secret_access_key(
        mut self,
        secret_access_key: impl Into<String>,
    ) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_secret_access_key(secret_access_key));
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_session_token(mut self, session_token: impl Into<String>) -> DfResult<Self> {
        self.s3 = Some(self.take_s3()?.with_token(session_token));
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_service_account(mut self, path: impl Into<String>) -> DfResult<Self> {
        self.gcs = Some(self.take_gcs()?.with_service_account_path(path));
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_service_account_key(mut self, key: impl Into<String>) -> DfResult<Self> {
        self.gcs = Some(self.take_gcs()?.with_service_account_key(key));
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_application_credentials(mut self, path: impl Into<String>) -> DfResult<Self> {
        self.gcs = Some(self.take_gcs()?.with_application_credentials(path));
        Ok(self)
    }

    // GCS has no dedicated endpoint setter; the base URL is set through the
    // stringly-typed config map (object_store 0.13's `GoogleConfigKey::BaseUrl`).
    #[cfg(feature = "gcs")]
    pub fn with_gcs_endpoint(mut self, endpoint: impl Into<String>) -> DfResult<Self> {
        self.gcs = Some(
            self.take_gcs()?
                .with_config(GoogleConfigKey::BaseUrl, endpoint),
        );
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_anonymous(mut self, anonymous: bool) -> DfResult<Self> {
        self.gcs = Some(self.take_gcs()?.with_skip_signature(anonymous));
        Ok(self)
    }

    // Resolve into a `ZarrTableUrl`, dispatching on the url scheme to
    // the matching backend.
    pub fn build(self) -> DfResult<ZarrTableUrl> {
        match self.url.scheme() {
            "file" => self.build_local(),
            #[cfg(feature = "s3")]
            "s3" => self.build_s3(),
            #[cfg(feature = "gcs")]
            "gs" => self.build_gcs(),
            other => Err(DataFusionError::Execution(format!(
                "unsupported url scheme '{other}' for a zarr store"
            ))),
        }
    }

    // Build a local (file) zarr store. Cloud options are invalid here.
    fn build_local(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "s3 options were provided for a local (file) zarr store".into(),
            ));
        }
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "gcs options were provided for a local (file) zarr store".into(),
            ));
        }
        Ok(ZarrTableUrl::ZarrStore { url: self.url })
    }

    #[cfg(feature = "s3")]
    fn build_s3(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "gcs options were provided for an s3 zarr store".into(),
            ));
        }
        let builder = self.s3.unwrap_or_else(AmazonS3Builder::from_env);
        Ok(ZarrTableUrl::S3Zarr {
            url: self.url,
            builder,
        })
    }

    #[cfg(feature = "gcs")]
    fn build_gcs(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "s3 options were provided for a gcs zarr store".into(),
            ));
        }
        let builder = self.gcs.unwrap_or_else(GoogleCloudStorageBuilder::from_env);
        Ok(ZarrTableUrl::GcsZarr {
            url: self.url,
            builder,
        })
    }
}

// Accumulator for icechunk s3 credentials. access_key_id +
// secret_access_key  are gathered here and resolved to a concrete
// `S3Credentials` at build time, erroring on partial/contradictory
// input.
#[cfg(all(feature = "icechunk", feature = "s3"))]
#[derive(Default)]
struct S3IcechunkBuilder {
    options: S3Options,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    anonymous: bool,
}

#[cfg(all(feature = "icechunk", feature = "s3"))]
impl S3IcechunkBuilder {
    fn build(self) -> DfResult<S3IcechunkOptions> {
        let credentials = if self.anonymous {
            if self.access_key_id.is_some()
                || self.secret_access_key.is_some()
                || self.session_token.is_some()
            {
                return Err(DataFusionError::Execution(
                    "s3.anonymous cannot be combined with s3.access_key_id / \
                     s3.secret_access_key / s3.session_token"
                        .into(),
                ));
            }
            S3Credentials::Anonymous
        } else {
            match (self.access_key_id, self.secret_access_key) {
                (Some(access_key_id), Some(secret_access_key)) => {
                    S3Credentials::Static(S3StaticCredentials {
                        access_key_id,
                        secret_access_key,
                        session_token: self.session_token,
                        expires_after: None,
                    })
                }
                (None, None) if self.session_token.is_none() => S3Credentials::FromEnv,
                _ => {
                    return Err(DataFusionError::Execution(
                        "s3.access_key_id and s3.secret_access_key must both be set \
                         (session_token requires both)"
                            .into(),
                    ))
                }
            }
        };
        Ok(S3IcechunkOptions {
            options: self.options,
            credentials,
        })
    }
}

// Parse an icechunk s3 checksum algorithm option.
#[cfg(all(feature = "icechunk", feature = "s3"))]
fn parse_s3_checksum(value: &str) -> DfResult<S3ChecksumAlgorithm> {
    match value {
        "crc32" => Ok(S3ChecksumAlgorithm::Crc32),
        "crc32c" => Ok(S3ChecksumAlgorithm::Crc32c),
        "crc64nvme" => Ok(S3ChecksumAlgorithm::Crc64Nvme),
        "sha1" => Ok(S3ChecksumAlgorithm::Sha1),
        "sha256" => Ok(S3ChecksumAlgorithm::Sha256),
        other => Err(DataFusionError::Execution(format!(
            "unknown s3 checksum algorithm '{other}' (expected one of: crc32, crc32c, \
             crc64nvme, sha1, sha256)"
        ))),
    }
}

// Builder for an icechunk-backed `ZarrTableUrl`.
//
// Mirrors `ZarrUrlBuilder` but for icechunk repos. As with the
// zarr builder, s3 and gcs options are mutually exclusive and the
// `build_*` methods reject options meant for a different backend.
// Namespaces: `icechunk.*` `s3.*` and `gcs.*`.
#[cfg(feature = "icechunk")]
pub struct IcechunkUrlBuilder {
    url: ListingTableUrl,
    icechunk: IcechunkConfig,
    // Tracks whether a version selector was already set, so a second
    // one errors instead of silently overwriting (the default version
    // stays otherwise).
    version_set: bool,
    #[cfg(feature = "s3")]
    s3: Option<S3IcechunkBuilder>,
    #[cfg(feature = "gcs")]
    gcs: Option<GcsCredentials>,
}

#[cfg(feature = "icechunk")]
impl IcechunkUrlBuilder {
    pub fn try_new(
        url: ListingTableUrl,
        options: Option<&HashMap<String, String>>,
    ) -> DfResult<Self> {
        let mut builder = Self {
            url,
            icechunk: IcechunkConfig::default(),
            version_set: false,
            #[cfg(feature = "s3")]
            s3: None,
            #[cfg(feature = "gcs")]
            gcs: None,
        };
        if let Some(options) = options {
            for (key, value) in options {
                builder = builder.apply_option(key, value)?;
            }
        }
        Ok(builder)
    }

    fn apply_option(self, key: &str, value: &str) -> DfResult<Self> {
        let (namespace, option) = key.split_once('.').ok_or_else(|| {
            DataFusionError::Execution(format!(
                "invalid option key '{key}': expected a namespaced key like 'icechunk.branch'"
            ))
        })?;
        match namespace {
            "icechunk" => self.apply_icechunk_option(option, value),
            #[cfg(feature = "s3")]
            "s3" => self.apply_s3_option(option, value),
            #[cfg(feature = "gcs")]
            "gcs" => self.apply_gcs_option(option, value),
            other => Err(DataFusionError::Execution(format!(
                "unknown or unsupported option namespace '{other}' for an icechunk repo"
            ))),
        }
    }

    fn apply_icechunk_option(self, option: &str, value: &str) -> DfResult<Self> {
        match option {
            "branch" => self.with_icechunk_version(IcechunkVersion::BranchTip(value.to_string())),
            "tag" => self.with_icechunk_version(IcechunkVersion::RefTag(value.to_string())),
            "snapshot_id" => {
                self.with_icechunk_version(IcechunkVersion::SnapshotId(value.to_string()))
            }
            "group" => self.with_icechunk_group(value),
            other => Err(DataFusionError::Execution(format!(
                "unknown icechunk option '{other}' (expected one of: branch, tag, snapshot_id, group)"
            ))),
        }
    }

    pub fn with_icechunk_group(mut self, group: impl Into<String>) -> DfResult<Self> {
        let group = group.into();
        let group = group.trim_matches('/');
        if group.is_empty() {
            return Err(DataFusionError::Execution(
                "icechunk.group must be a non-empty path".into(),
            ));
        }
        self.icechunk.group = Some(group.to_string());
        Ok(self)
    }

    pub fn with_icechunk_version(mut self, version: IcechunkVersion) -> DfResult<Self> {
        if self.version_set {
            return Err(DataFusionError::Execution(
                "at most one of icechunk.branch / icechunk.tag / icechunk.snapshot_id may be set"
                    .into(),
            ));
        }
        self.icechunk.version = version;
        self.version_set = true;
        Ok(self)
    }

    #[cfg(feature = "s3")]
    fn apply_s3_option(self, option: &str, value: &str) -> DfResult<Self> {
        match option {
            "region" => self.with_s3_region(value),
            "endpoint" => self.with_s3_endpoint(value),
            "allow_http" => self.with_s3_allow_http(parse_bool_option(option, value)?),
            "force_path_style" => self.with_s3_force_path_style(parse_bool_option(option, value)?),
            "anonymous" => self.with_s3_anonymous(parse_bool_option(option, value)?),
            "requester_pays" => self.with_s3_requester_pays(parse_bool_option(option, value)?),
            "checksum_algorithm" => self.with_s3_checksum_algorithm(parse_s3_checksum(value)?),
            "network_stream_timeout_seconds" => {
                let seconds = value.parse::<u32>().map_err(|_| {
                    DataFusionError::Execution(format!(
                        "invalid integer value '{value}' for option 'network_stream_timeout_seconds'"
                    ))
                })?;
                self.with_s3_network_stream_timeout_seconds(seconds)
            }
            "access_key_id" => self.with_s3_access_key_id(value),
            "secret_access_key" => self.with_s3_secret_access_key(value),
            "session_token" => self.with_s3_session_token(value),
            other => Err(DataFusionError::Execution(format!(
                "unknown s3 option '{other}' for an icechunk repo"
            ))),
        }
    }

    #[cfg(feature = "gcs")]
    fn apply_gcs_option(self, option: &str, value: &str) -> DfResult<Self> {
        match option {
            "service_account" => self.with_gcs_service_account(value),
            "service_account_key" => self.with_gcs_service_account_key(value),
            "application_credentials" => self.with_gcs_application_credentials(value),
            "bearer_token" => self.with_gcs_bearer_token(value),
            "anonymous" => self.with_gcs_anonymous(parse_bool_option(option, value)?),
            other => Err(DataFusionError::Execution(format!(
                "unknown gcs option '{other}' for an icechunk repo"
            ))),
        }
    }

    #[cfg(feature = "s3")]
    fn take_s3(&mut self) -> DfResult<S3IcechunkBuilder> {
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "cannot set both s3 and gcs options on the same icechunk repo".into(),
            ));
        }
        Ok(self.s3.take().unwrap_or_default())
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_region(mut self, region: impl Into<String>) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_region(region);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_endpoint(mut self, endpoint: impl Into<String>) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_endpoint_url(endpoint);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_allow_http(mut self, allow_http: bool) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_allow_http(allow_http);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_force_path_style(mut self, force_path_style: bool) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_force_path_style(force_path_style);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_anonymous(mut self, anonymous: bool) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.anonymous = anonymous;
        // icechunk applies `S3Options::anonymous` after the credentials, so it
        // must agree with `S3Credentials::Anonymous` or it overrides it.
        b.options = b.options.with_anonymous(anonymous);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_requester_pays(mut self, requester_pays: bool) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_requester_pays(requester_pays);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_checksum_algorithm(mut self, algorithm: S3ChecksumAlgorithm) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_checksum_algorithm(algorithm);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_network_stream_timeout_seconds(mut self, seconds: u32) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.options = b.options.with_network_stream_timeout_seconds(seconds);
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_access_key_id(mut self, access_key_id: impl Into<String>) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.access_key_id = Some(access_key_id.into());
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_secret_access_key(
        mut self,
        secret_access_key: impl Into<String>,
    ) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.secret_access_key = Some(secret_access_key.into());
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "s3")]
    pub fn with_s3_session_token(mut self, session_token: impl Into<String>) -> DfResult<Self> {
        let mut b = self.take_s3()?;
        b.session_token = Some(session_token.into());
        self.s3 = Some(b);
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    fn set_gcs_credentials(mut self, credentials: GcsCredentials) -> DfResult<Self> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "cannot set both s3 and gcs options on the same icechunk repo".into(),
            ));
        }
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "at most one gcs credential source may be set".into(),
            ));
        }
        self.gcs = Some(credentials);
        Ok(self)
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_service_account(self, path: impl Into<String>) -> DfResult<Self> {
        self.set_gcs_credentials(GcsCredentials::Static(
            GcsStaticCredentials::ServiceAccount(PathBuf::from(path.into())),
        ))
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_service_account_key(self, key: impl Into<String>) -> DfResult<Self> {
        self.set_gcs_credentials(GcsCredentials::Static(
            GcsStaticCredentials::ServiceAccountKey(key.into()),
        ))
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_application_credentials(self, path: impl Into<String>) -> DfResult<Self> {
        self.set_gcs_credentials(GcsCredentials::Static(
            GcsStaticCredentials::ApplicationCredentials(PathBuf::from(path.into())),
        ))
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_bearer_token(self, token: impl Into<String>) -> DfResult<Self> {
        self.set_gcs_credentials(GcsCredentials::Static(GcsStaticCredentials::BearerToken(
            GcsBearerCredential {
                bearer: token.into(),
                expires_after: None,
            },
        )))
    }

    #[cfg(feature = "gcs")]
    pub fn with_gcs_anonymous(self, anonymous: bool) -> DfResult<Self> {
        if anonymous {
            self.set_gcs_credentials(GcsCredentials::Anonymous)
        } else {
            Ok(self)
        }
    }

    // Resolve into a `ZarrTableUrl`, dispatching on the url scheme to the
    // matching backend.
    pub fn build(self) -> DfResult<ZarrTableUrl> {
        match self.url.scheme() {
            "file" => self.build_local(),
            #[cfg(feature = "s3")]
            "s3" => self.build_s3(),
            #[cfg(feature = "gcs")]
            "gs" => self.build_gcs(),
            other => Err(DataFusionError::Execution(format!(
                "unsupported url scheme '{other}' for an icechunk repo"
            ))),
        }
    }

    // Build a local (file) icechunk repo. Cloud options are invalid here.
    fn build_local(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "s3 options were provided for a local (file) icechunk repo".into(),
            ));
        }
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "gcs options were provided for a local (file) icechunk repo".into(),
            ));
        }
        Ok(ZarrTableUrl::IcechunkRepo {
            url: self.url,
            icechunk: self.icechunk,
        })
    }

    #[cfg(feature = "s3")]
    fn build_s3(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "gcs")]
        if self.gcs.is_some() {
            return Err(DataFusionError::Execution(
                "gcs options were provided for an s3 icechunk repo".into(),
            ));
        }
        let s3 = match self.s3 {
            Some(b) => b.build()?,
            None => S3IcechunkOptions::default(),
        };
        Ok(ZarrTableUrl::S3Icechunk {
            url: self.url,
            icechunk: self.icechunk,
            s3,
        })
    }

    #[cfg(feature = "gcs")]
    fn build_gcs(self) -> DfResult<ZarrTableUrl> {
        #[cfg(feature = "s3")]
        if self.s3.is_some() {
            return Err(DataFusionError::Execution(
                "s3 options were provided for a gcs icechunk repo".into(),
            ));
        }
        Ok(ZarrTableUrl::GcsIcechunk {
            url: self.url,
            icechunk: self.icechunk,
            credentials: self.gcs.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod zarr_config_tests {
    use super::*;
    #[cfg(feature = "icechunk")]
    use crate::test_utils::get_local_icechunk_repo;
    use crate::test_utils::{get_local_zarr_store, get_local_zarr_store_mix_dims};

    #[tokio::test]
    async fn schema_inference_tests() {
        // local zarr directory.
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "data_for_config_dir").await;
        let path = wrapper.get_store_path();

        let table_url = ListingTableUrl::parse(path).unwrap();
        let zarr_table_url = ZarrUrlBuilder::try_new(table_url, None)
            .unwrap()
            .build_local()
            .unwrap();
        let (inferred_schema, _stats) = zarr_table_url.infer_schema().await.unwrap();
        assert_eq!(inferred_schema, schema);

        // local icechunk repo.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, schema) =
                get_local_icechunk_repo(true, 0.0, "data_for_config_repo").await;
            let path = wrapper.get_store_path();

            let table_url = ListingTableUrl::parse(path).unwrap();
            let zarr_table_url = IcechunkUrlBuilder::try_new(table_url, None)
                .unwrap()
                .build_local()
                .unwrap();
            let (inferred_schema, _stats) = zarr_table_url.infer_schema().await.unwrap();
            assert_eq!(inferred_schema, schema);
        }
    }

    #[tokio::test]
    async fn statistics_tests() {
        // local zarr directory with mixed-dimension coordinates: `lat` is a 2D (8x8)
        // broadcasted coordinate, `lon` is 1D (8), and `data` is 2D (8x8). so `lat` and
        // `data` span both dimensions (64 rows), while `lon` spans a single one (8 rows).
        let (wrapper, schema) = get_local_zarr_store_mix_dims(0.0, "data_for_config_stats").await;
        let path = wrapper.get_store_path();
        let table_url = ZarrUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
            .unwrap()
            .build_local()
            .unwrap();
        let (inferred_schema, stats_base) = table_url.infer_schema().await.unwrap();
        assert_eq!(inferred_schema, schema);

        let base =
            ZarrTableConfig::new(table_url, inferred_schema.clone()).with_stats_base(stats_base);
        let num_rows = |cols: &[&str]| {
            let projection: Vec<usize> = cols
                .iter()
                .map(|c| inferred_schema.index_of(c).unwrap())
                .collect();
            base.clone()
                .with_projection(projection)
                .statistics()
                .unwrap()
                .num_rows
        };

        assert_eq!(num_rows(&["lat"]), Precision::Exact(64));
        assert_eq!(num_rows(&["lon"]), Precision::Exact(8));
        assert_eq!(num_rows(&["data"]), Precision::Exact(64));
        assert_eq!(num_rows(&["data", "lat", "lon"]), Precision::Exact(64));

        // local icechunk repo with 1D `lat`/`lon` coordinates and 2D `data`. selecting
        // both coordinates broadcasts them to the full 2D grid -> 64 rows.
        #[cfg(feature = "icechunk")]
        {
            let (wrapper, _schema) =
                get_local_icechunk_repo(true, 0.0, "data_for_config_stats_repo").await;
            let path = wrapper.get_store_path();
            let table_url =
                IcechunkUrlBuilder::try_new(ListingTableUrl::parse(path).unwrap(), None)
                    .unwrap()
                    .build_local()
                    .unwrap();
            let (inferred_schema, stats_base) = table_url.infer_schema().await.unwrap();

            let projection = vec![
                inferred_schema.index_of("lat").unwrap(),
                inferred_schema.index_of("lon").unwrap(),
            ];
            let config = ZarrTableConfig::new(table_url, inferred_schema)
                .with_stats_base(stats_base)
                .with_projection(projection);
            assert_eq!(config.statistics().unwrap().num_rows, Precision::Exact(64));
        }
    }

    // Builds a `ZarrTableUrl` straight from the builder (no store / no
    // schema inference).
    #[cfg(all(feature = "s3", feature = "gcs"))]
    #[test]
    fn zarr_url_builder_options_tests() {
        // s3 and gcs options are mutually exclusive -> error (in either apply order).
        let mixed = HashMap::from([
            ("s3.region".to_string(), "us-east-1".to_string()),
            ("gcs.anonymous".to_string(), "true".to_string()),
        ]);
        assert!(ZarrUrlBuilder::try_new(
            ListingTableUrl::parse("s3://bucket/store").unwrap(),
            Some(&mixed)
        )
        .is_err());

        // valid s3 options build an s3 zarr store.
        let s3_opts = HashMap::from([
            ("s3.region".to_string(), "us-east-1".to_string()),
            (
                "s3.endpoint".to_string(),
                "http://localhost:9000".to_string(),
            ),
            ("s3.allow_http".to_string(), "true".to_string()),
        ]);
        let built = ZarrUrlBuilder::try_new(
            ListingTableUrl::parse("s3://bucket/store").unwrap(),
            Some(&s3_opts),
        )
        .unwrap()
        .build()
        .unwrap();
        assert!(matches!(built, ZarrTableUrl::S3Zarr { .. }));

        // valid gcs options build a gcs zarr store.
        let gcs_opts = HashMap::from([("gcs.anonymous".to_string(), "true".to_string())]);
        let built = ZarrUrlBuilder::try_new(
            ListingTableUrl::parse("gs://bucket/store").unwrap(),
            Some(&gcs_opts),
        )
        .unwrap()
        .build()
        .unwrap();
        assert!(matches!(built, ZarrTableUrl::GcsZarr { .. }));
    }

    // The icechunk builder resolves user options into icechunk's own typed
    // `S3Options` / `S3Credentials` / `GcsCredentials` without ever reading the
    // environment, so we can assert on the resolved config directly.
    #[cfg(all(feature = "icechunk", feature = "s3", feature = "gcs"))]
    #[test]
    fn icechunk_url_builder_options_tests() {
        // s3 and gcs options are mutually exclusive -> error.
        let mixed = HashMap::from([
            ("s3.region".to_string(), "us-east-1".to_string()),
            ("gcs.anonymous".to_string(), "true".to_string()),
        ]);
        assert!(IcechunkUrlBuilder::try_new(
            ListingTableUrl::parse("s3://bucket/repo").unwrap(),
            Some(&mixed)
        )
        .is_err());

        // valid s3 options (+ a version selector) build an s3 icechunk repo,
        // and the resolved icechunk config carries exactly what was requested.
        let s3_opts = HashMap::from([
            ("s3.region".to_string(), "us-west-2".to_string()),
            ("s3.access_key_id".to_string(), "AKIA".to_string()),
            ("s3.secret_access_key".to_string(), "secret".to_string()),
            ("icechunk.branch".to_string(), "dev".to_string()),
        ]);
        let built = IcechunkUrlBuilder::try_new(
            ListingTableUrl::parse("s3://bucket/repo").unwrap(),
            Some(&s3_opts),
        )
        .unwrap()
        .build()
        .unwrap();
        let ZarrTableUrl::S3Icechunk { s3, icechunk, .. } = built else {
            panic!("expected an s3 icechunk repo");
        };
        assert_eq!(s3.options.region.as_deref(), Some("us-west-2"));
        assert!(
            matches!(&s3.credentials, S3Credentials::Static(c) if c.access_key_id == "AKIA"
                && c.secret_access_key == "secret")
        );
        assert!(matches!(icechunk.version, IcechunkVersion::BranchTip(ref b) if b == "dev"));

        // valid gcs options build a gcs icechunk repo.
        let gcs_opts = HashMap::from([("gcs.anonymous".to_string(), "true".to_string())]);
        let built = IcechunkUrlBuilder::try_new(
            ListingTableUrl::parse("gs://bucket/repo").unwrap(),
            Some(&gcs_opts),
        )
        .unwrap()
        .build()
        .unwrap();
        assert!(matches!(built, ZarrTableUrl::GcsIcechunk { .. }));
    }
}
