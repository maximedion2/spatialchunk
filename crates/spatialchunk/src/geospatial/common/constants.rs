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

use arrow_schema::DataType;
use datafusion::common::DataFusionError;

/// The Arrow field-metadata key for an extension type's name.
pub(crate) const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// The Arrow field-metadata key for an extension type's metadata. For the
/// geoarrow extensions it's a JSON object with the crs and the edges, see
/// `GeoMetadata`.
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

// The storage types of a wkb geo column.
pub(crate) const WKB_DATA_TYPES: [DataType; 2] = [DataType::Binary, DataType::BinaryView];

pub(crate) const EARTH_RADIUS_METERS: f64 = 6_371_010.0;

// The longitudes of a lon/lat crs, (min, max).
pub(crate) const LON_RANGE: (f64, f64) = (-180.0, 180.0);

// Cheap geometry-validity errors caught at construction (not full
// DE-9IM validation, just the checks that prevent later misbehavior
// like divide-by-zero).
#[derive(Debug)]
pub(crate) enum GeoError {
    // The WKB bytes could not be parsed.
    MalformedWkb(String),
    // A segment/edge with identical endpoints (zero length).
    DegenerateSegment,
    // A spherical segment/edge with (near) antipodal endpoints, for which the
    // great circle is undefined.
    AntipodalSegment,
    // A polygon ring whose first and last vertices differ (or that
    // produced no edges).
    UnclosedRing,
    // A spatial join was given JoinSide::None, which has no valid operand ordering.
    InvalidJoinSide,
}

impl std::fmt::Display for GeoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoError::MalformedWkb(msg) => write!(f, "malformed WKB: {msg}"),
            GeoError::DegenerateSegment => write!(f, "degenerate segment (identical endpoints)"),
            GeoError::AntipodalSegment => {
                write!(f, "antipodal segment (great circle is undefined)")
            }
            GeoError::UnclosedRing => write!(f, "polygon ring is not closed"),
            GeoError::InvalidJoinSide => {
                write!(f, "JoinSide::None is not valid for a spatial join")
            }
        }
    }
}

impl std::error::Error for GeoError {}

pub(crate) fn geo_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}
