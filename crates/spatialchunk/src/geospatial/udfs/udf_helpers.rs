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

//! Helpers shared by the geo udfs (and the readers that label geo columns):
//! the geoarrow extension labels on the fields, i.e. the extension name and
//! the extension metadata with the crs and edges.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{exec_datafusion_err, plan_err, DataFusionError, Result, ScalarValue};
use datafusion::logical_expr::{ReturnFieldArgs, Signature, TypeSignature, Volatility};
use proj4rs::Proj;
use serde_json::{Map, Value};

/// The Arrow field-metadata key for an extension type's name.
pub(crate) const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// The Arrow field-metadata key for an extension type's metadata. For the
/// geoarrow extensions it's a JSON object with the crs and the edges, see
/// [`GeoMetadata`].
pub(crate) const ARROW_EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";

/// The GeoArrow extension name that marks a binary column as WKB geometry. A
/// column carrying this on its field metadata is treated as geometry; a bare
/// binary column is not.
pub(crate) const GEOARROW_WKB_EXTENSION_NAME: &str = "geoarrow.wkb";

/// The crs of geoparquet columns that don't specify one. the spec defaults to
/// OGC:CRS84, i.e. EPSG:4326 in lon/lat order.
pub(crate) const DEFAULT_GEOPARQUET_CRS: &str = "EPSG:4326";

pub(crate) const GEOGRAPHY_CRS: &str = "EPSG:4326";
pub(crate) const GEOGRAPHY_EDGES: &str = "spherical";

// The geoarrow extension metadata of a geo column, stored as JSON under
// `ARROW:extension:metadata`, e.g. `{"crs": "EPSG:4326", "edges": "spherical"}`.
//
// - `crs` is saved as provided (a string, or a PROJJSON object from
//   geoparquet), it's not parsed or checked here. None means unknown.
// - `edges` is None for planar edges, "spherical" for a geography.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GeoMetadata {
    pub(crate) crs: Option<Value>,
    pub(crate) edges: Option<String>,
}

impl GeoMetadata {
    pub(crate) fn with_crs(crs: Option<Value>) -> Self {
        Self { crs, edges: None }
    }

    pub(crate) fn is_geography(&self) -> bool {
        self.edges.is_some()
    }

    pub(crate) fn to_json(&self) -> String {
        let mut json = Map::new();
        if let Some(crs) = &self.crs {
            json.insert("crs".to_string(), crs.clone());
        }
        if let Some(edges) = &self.edges {
            json.insert("edges".to_string(), Value::String(edges.clone()));
        }
        Value::Object(json).to_string()
    }

    // The metadata of a field, empty if it has none. "planar" edges are the
    // same as no edges, and a null crs is the same as no crs.
    pub(crate) fn from_field(field: &Field) -> Result<Self> {
        let Some(json) = field.metadata().get(ARROW_EXTENSION_METADATA_KEY) else {
            return Ok(Self::default());
        };
        let json: Value = serde_json::from_str(json).map_err(|e| {
            exec_datafusion_err!(
                "invalid {ARROW_EXTENSION_METADATA_KEY} on {}: {e}",
                field.name()
            )
        })?;
        let crs = json.get("crs").filter(|crs| !crs.is_null()).cloned();
        let edges = json
            .get("edges")
            .and_then(Value::as_str)
            .map(str::to_lowercase)
            .filter(|edges| edges != "planar");
        Ok(Self { crs, edges })
    }
}

// Builds an output `Field` for a geometry column: the given storage type tagged
// with the `geoarrow.wkb` extension name so downstream operations recognize it,
// and with the crs/edges in the extension metadata.
pub(crate) fn geometry_field(
    name: &str,
    storage_type: DataType,
    nullable: bool,
    geo_metadata: &GeoMetadata,
) -> FieldRef {
    let metadata = HashMap::from([
        (
            ARROW_EXTENSION_NAME_KEY.to_string(),
            GEOARROW_WKB_EXTENSION_NAME.to_string(),
        ),
        (
            ARROW_EXTENSION_METADATA_KEY.to_string(),
            geo_metadata.to_json(),
        ),
    ]);
    Arc::new(Field::new(name, storage_type, nullable).with_metadata(metadata))
}

// Errors unless `field` is tagged as WKB geometry (the `geoarrow.wkb` extension
// name). Used to reject bare binary columns from geometry UDFs and the spatial
// join, forcing an explicit `st_geomfromwkb` / `st_geomfromwkt` tag first.
pub(crate) fn require_geometry(field: &Field) -> Result<()> {
    match field.metadata().get(ARROW_EXTENSION_NAME_KEY) {
        Some(name) if name == GEOARROW_WKB_EXTENSION_NAME => Ok(()),
        _ => plan_err!(
            "expected a geometry column tagged '{GEOARROW_WKB_EXTENSION_NAME}' \
             (use st_geomfromwkb / st_geomfromwkt), got a plain {} column",
            field.data_type()
        ),
    }
}

// Errors unless `field` is a geometry, i.e. tagged as WKB geometry and not a
// geography. returns its crs/edges.
pub(crate) fn require_planar_geometry(udf_name: &str, field: &Field) -> Result<GeoMetadata> {
    require_geometry(field)?;
    let metadata = GeoMetadata::from_field(field)?;
    if metadata.is_geography() {
        return plan_err!("{udf_name}: expected a geometry, got a geography");
    }
    Ok(metadata)
}

/// The signature of a udf with an optional trailing crs argument: each of
/// `args` alone, then followed by a string or an integer crs.
pub(crate) fn signature_with_optional_crs(args: Vec<Vec<DataType>>) -> Signature {
    crs_signature(args, true)
}

/// The signature of a udf with a trailing crs argument: each of `args`
/// followed by a string or an integer crs.
pub(crate) fn signature_with_crs(args: Vec<Vec<DataType>>) -> Signature {
    crs_signature(args, false)
}

fn crs_signature(args: Vec<Vec<DataType>>, optional: bool) -> Signature {
    let mut crs_types = vec![Some(DataType::Utf8), Some(DataType::Int64)];
    if optional {
        crs_types.insert(0, None);
    }
    let mut signatures = Vec::new();
    for args in args {
        for crs_type in &crs_types {
            let mut args = args.clone();
            args.extend(crs_type.clone());
            signatures.push(TypeSignature::Exact(args));
        }
    }
    Signature::one_of(signatures, Volatility::Immutable)
}

// A bare code like "4326" is the same as "EPSG:4326", anything else is kept as
// is.
pub(crate) fn normalized_crs(crs: &Value) -> Value {
    match crs.as_str().map(str::trim) {
        Some(code) if code.parse::<u16>().is_ok() => Value::String(format!("EPSG:{code}")),
        Some(crs) => Value::String(crs.to_string()),
        None => crs.clone(),
    }
}

// The crs is what proj4rs supports: "EPSG:<code>", "WGS84" or a proj string,
// after normalization.
pub(crate) fn proj_from_crs(udf_name: &str, crs: &Value) -> Result<Proj> {
    let normalized = normalized_crs(crs);
    let Some(user_string) = normalized.as_str() else {
        return plan_err!("{udf_name}: unsupported crs {crs}");
    };
    Proj::from_user_string(user_string)
        .map_err(|e| DataFusionError::Plan(format!("{udf_name}: unsupported crs {crs}: {e}")))
}

/// The crs argument at `crs_idx`, if the udf was called with one, e.g. the
/// 'EPSG:4326' of st_point(x, y, 'EPSG:4326'). it ends up on the output field,
/// which is fixed at planning, so it must be a literal.
pub(crate) fn crs_argument(
    udf_name: &str,
    args: &ReturnFieldArgs,
    crs_idx: usize,
) -> Result<Option<Value>> {
    if args.arg_fields.len() <= crs_idx {
        return Ok(None);
    }
    let Some(Some(crs)) = args.scalar_arguments.get(crs_idx) else {
        return plan_err!("{udf_name}: the crs must be a literal, e.g. 'EPSG:4326' or 4326");
    };
    if crs.is_null() {
        return Ok(None);
    }
    match crs {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Ok(Some(Value::String(s.clone()))),
        ScalarValue::Int8(_)
        | ScalarValue::Int16(_)
        | ScalarValue::Int32(_)
        | ScalarValue::Int64(_)
        | ScalarValue::UInt8(_)
        | ScalarValue::UInt16(_)
        | ScalarValue::UInt32(_)
        | ScalarValue::UInt64(_) => Ok(Some(Value::String(crs.to_string()))),
        _ => plan_err!(
            "{udf_name}: the crs must be a string or an integer, got {}",
            crs.data_type()
        ),
    }
}

#[cfg(test)]
mod udf_helpers_tests {
    use super::*;

    fn return_field_args<'a>(
        fields: &'a [FieldRef],
        scalars: &'a [Option<&'a ScalarValue>],
    ) -> ReturnFieldArgs<'a> {
        ReturnFieldArgs {
            arg_fields: fields,
            scalar_arguments: scalars,
        }
    }

    #[test]
    fn geo_metadata_round_trips_through_a_field() {
        let metadata = GeoMetadata {
            crs: Some(Value::String("EPSG:4326".into())),
            edges: Some("spherical".into()),
        };
        let field = geometry_field("g", DataType::Binary, true, &metadata);
        assert_eq!(GeoMetadata::from_field(&field).unwrap(), metadata);
        assert!(metadata.is_geography());

        // nothing set is an empty object.
        let field = geometry_field("g", DataType::Binary, true, &GeoMetadata::default());
        assert_eq!(
            field
                .metadata()
                .get(ARROW_EXTENSION_METADATA_KEY)
                .map(String::as_str),
            Some("{}")
        );
        assert_eq!(
            GeoMetadata::from_field(&field).unwrap(),
            GeoMetadata::default()
        );
    }

    #[test]
    fn geo_metadata_planar_edges_and_null_crs_are_unset() {
        let field = Field::new("g", DataType::Binary, true).with_metadata(HashMap::from([(
            ARROW_EXTENSION_METADATA_KEY.to_string(),
            r#"{"crs": null, "edges": "planar"}"#.to_string(),
        )]));
        assert_eq!(
            GeoMetadata::from_field(&field).unwrap(),
            GeoMetadata::default()
        );
    }

    #[test]
    fn crs_argument_literals() {
        let fields: Vec<FieldRef> = vec![
            Arc::new(Field::new("g", DataType::Binary, true)),
            Arc::new(Field::new("crs", DataType::Utf8, true)),
        ];
        let crs = |scalar: ScalarValue| {
            crs_argument("f", &return_field_args(&fields, &[None, Some(&scalar)]), 1)
        };

        assert_eq!(
            crs(ScalarValue::from("EPSG:4326")).unwrap(),
            Some(Value::String("EPSG:4326".into()))
        );
        assert_eq!(
            crs(ScalarValue::Int64(Some(4326))).unwrap(),
            Some(Value::String("4326".into()))
        );
        assert_eq!(crs(ScalarValue::Utf8(None)).unwrap(), None);
        assert!(crs(ScalarValue::Float64(Some(1.0))).is_err());

        // not a literal.
        assert!(crs_argument("f", &return_field_args(&fields, &[None, None]), 1).is_err());
        // no crs argument.
        assert_eq!(
            crs_argument("f", &return_field_args(&fields[..1], &[None]), 1).unwrap(),
            None
        );
    }
}
