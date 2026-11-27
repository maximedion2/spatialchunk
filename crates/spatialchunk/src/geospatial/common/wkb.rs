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
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{exec_datafusion_err, exec_err, Result};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
use geo_types::{LineString, Point};
use serde_json::{Map, Value};
use wkb::reader::Wkb;
use wkb::writer::{write_geometry, WriteOptions};

use super::constants::{
    ARROW_EXTENSION_METADATA_KEY, ARROW_EXTENSION_NAME_KEY, GEOARROW_WKB_EXTENSION_NAME,
};

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

// A Binary wkb field with the crs/edges of `field`, which can have any geo
// encoding, e.g. a geoarrow.polygon column.
pub(crate) fn wkb_field_like(field: &Field) -> Result<FieldRef> {
    let metadata = GeoMetadata::from_field(field)?;
    Ok(geometry_field(
        field.name(),
        DataType::Binary,
        true,
        &metadata,
    ))
}

// Reads the WKB bytes at `row_idx`, or `None` if the row is null. The signature
// restricts geometry inputs to Binary / BinaryView.
pub(crate) fn wkb_at(array: &ArrayRef, row_idx: usize) -> Option<&[u8]> {
    match array.data_type() {
        DataType::Binary => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("geometry array is Binary");
            (!a.is_null(row_idx)).then(|| a.value(row_idx))
        }
        DataType::BinaryView => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .expect("geometry array is BinaryView");
            (!a.is_null(row_idx)).then(|| a.value(row_idx))
        }
        other => panic!("unsupported geometry array type: {other}"),
    }
}

// The coordinates of the point at `row`, None if it's null or empty. errors if
// it's not a point.
pub(crate) fn point_at(wkb: &ArrayRef, row: usize) -> Result<Option<(f64, f64)>> {
    let Some(bytes) = wkb_at(wkb, row) else {
        return Ok(None);
    };
    let wkb = Wkb::try_new(bytes).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
    match wkb.as_type() {
        GeometryType::Point(p) => Ok(p.coord().map(|c| (c.x(), c.y()))),
        _ => exec_err!("expected a point"),
    }
}

pub(crate) fn point_wkb(x: f64, y: f64) -> Result<Vec<u8>> {
    to_wkb(&Point::new(x, y))
}

pub(crate) fn line_wkb(points: &[(f64, f64)]) -> Result<Vec<u8>> {
    to_wkb(&LineString::from(points.to_vec()))
}

fn to_wkb(geom: &impl GeometryTrait<T = f64>) -> Result<Vec<u8>> {
    let mut wkb = Vec::new();
    write_geometry(&mut wkb, geom, &WriteOptions::default())
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    Ok(wkb)
}

#[cfg(test)]
mod wkb_tests {
    use super::*;

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
}
