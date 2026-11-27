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

use datafusion::common::{plan_err, DataFusionError, Result};
use proj4rs::errors::Error as ProjError;
use proj4rs::transform::transform;
use proj4rs::Proj;
use serde_json::Value;

use super::constants::GEOGRAPHY_CRS;

pub(crate) fn geography_crs() -> Value {
    Value::String(GEOGRAPHY_CRS.to_string())
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
pub(crate) fn proj_from_crs(context: &str, crs: &Value) -> Result<Proj> {
    let normalized = normalized_crs(crs);
    let Some(user_string) = normalized.as_str() else {
        return plan_err!("{context}: unsupported crs {crs}");
    };
    Proj::from_user_string(user_string)
        .map_err(|e| DataFusionError::Plan(format!("{context}: unsupported crs {crs}: {e}")))
}

// A point transformed from `from` to `to`, in degrees for a lon/lat crs.
pub(crate) fn transform_point(
    from: &Proj,
    to: &Proj,
    x: f64,
    y: f64,
    z: f64,
) -> std::result::Result<(f64, f64, f64), ProjError> {
    // proj4rs works in radians for lon/lat.
    let mut point = if from.is_latlong() {
        (x.to_radians(), y.to_radians(), z)
    } else {
        (x, y, z)
    };
    transform(from, to, &mut point)?;
    // a point the projection isn't defined at doesn't always error.
    if point.0.is_nan() || point.1.is_nan() {
        return Err(ProjError::NanCoordinateValue);
    }
    if point.0.is_infinite() || point.1.is_infinite() {
        return Err(ProjError::CoordinateOutOfRange);
    }
    Ok(if to.is_latlong() {
        (point.0.to_degrees(), point.1.to_degrees(), point.2)
    } else {
        point
    })
}

pub(crate) fn is_latitude_valid(y: f64) -> bool {
    (-90.0..=90.0).contains(&y)
}

pub(crate) fn clamp_latitude(y: f64) -> f64 {
    y.clamp(-90.0, 90.0)
}

// A longitude wrapped to [-180, 180).
pub(crate) fn shift_longitude(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}

// Wraps a longitude difference into (-180, 180].
pub(crate) fn wrap_lon_diff(d: f64) -> f64 {
    if d > 180.0 {
        d - 360.0
    } else if d <= -180.0 {
        d + 360.0
    } else {
        d
    }
}
