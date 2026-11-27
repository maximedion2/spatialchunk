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

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};

/// `st_area(geo)` returns the area of a geo as a `Float64`, row-wise over a WKB
/// column: in the crs units squared for geometries, in square meters for
/// geographies.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StAreaUdf {
    signature: Signature,
}

impl Default for StAreaUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::BinaryView]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for StAreaUdf {
    fn name(&self) -> &str {
        "st_area"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        GeoType::from_field(self.name(), args.arg_fields[0].as_ref())?;
        Ok(Arc::new(Field::new(self.name(), DataType::Float64, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let geos = RawGeo::try_new(args.arg_fields[0].as_ref(), Arc::clone(&arrays[0]))?;
        Ok(ColumnarValue::Array(geos.area()?))
    }
}

#[cfg(test)]
mod test_helpers {
    use std::sync::Arc;

    use arrow::array::AsArray;
    use arrow::datatypes::Float64Type;
    use arrow_array::BinaryArray;
    use arrow_schema::{DataType, Field, FieldRef};
    use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
    use geos::Geom;
    use serde_json::Value;

    pub(super) use super::super::super::test_utils::wkt_to_wkb;
    use super::StAreaUdf;
    use crate::geospatial::geos::raw_geo::GeoType;

    pub(super) fn geometry() -> FieldRef {
        GeoType::geometry(None).to_field("geo", DataType::Binary, true)
    }

    pub(super) fn geography() -> FieldRef {
        GeoType::geometry(Some(Value::String("EPSG:4326".into())))
            .to_geography("test")
            .unwrap()
            .to_field("geo", DataType::Binary, true)
    }

    // The area of a single geo, through the udf.
    pub(super) fn area(field: FieldRef, wkt: &str) -> f64 {
        let bytes = wkt_to_wkb(wkt);
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(
                BinaryArray::from_iter_values([bytes]),
            ))],
            arg_fields: vec![field],
            number_rows: 1,
            return_field: Arc::new(Field::new("st_area", DataType::Float64, true)),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };
        match StAreaUdf::default().invoke_with_args(args).unwrap() {
            ColumnarValue::Array(a) => a.as_primitive::<Float64Type>().value(0),
            _ => panic!("expected array"),
        }
    }

    // Checks st_area against GEOS's area.
    pub(super) fn compare_area(label: &str, wkt: &str) {
        let geos_result = geos::Geometry::new_from_wkb(&wkt_to_wkb(wkt))
            .unwrap()
            .area()
            .unwrap();
        assert!(
            (geos_result - area(geometry(), wkt)).abs() < 1e-6,
            "{label}"
        );
    }
}

#[cfg(test)]
mod area_tests {
    use super::test_helpers::compare_area;

    // Vertices of a regular decagon of radius 10 centered at the origin.
    const DECAGON: [(f64, f64); 10] = [
        (10.0, 0.0),
        (8.0902, 5.8779),
        (3.0902, 9.5106),
        (-3.0902, 9.5106),
        (-8.0902, 5.8779),
        (-10.0, 0.0),
        (-8.0902, -5.8779),
        (-3.0902, -9.5106),
        (3.0902, -9.5106),
        (8.0902, -5.8779),
    ];

    // A decagon ring (closed) as a WKT coordinate list, shifted by (dx, dy).
    fn decagon_ring(dx: f64, dy: f64) -> String {
        DECAGON
            .iter()
            .chain(std::iter::once(&DECAGON[0]))
            .map(|(x, y)| format!("{} {}", x + dx, y + dy))
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[test]
    fn decagon() {
        compare_area(
            "decagon",
            &format!("POLYGON (({}))", decagon_ring(0.0, 0.0)),
        );
    }

    #[test]
    fn multipolygon() {
        compare_area(
            "two non-overlapping decagons",
            &format!(
                "MULTIPOLYGON ((({})), (({})))",
                decagon_ring(0.0, 0.0),
                decagon_ring(100.0, 0.0),
            ),
        );
    }

    #[test]
    fn decagon_with_hole() {
        compare_area(
            "decagon with square hole",
            &format!(
                "POLYGON (({}), (-2 -2, 2 -2, 2 2, -2 2, -2 -2))",
                decagon_ring(0.0, 0.0),
            ),
        );
    }
}

#[cfg(test)]
mod geography_area_tests {
    use s2::latlng::LatLng;
    use s2::point::{point_area, Point};

    use super::test_helpers::{area, geography};
    use crate::geospatial::common::constants::EARTH_RADIUS_METERS;

    fn s2_point(lon: f64, lat: f64) -> Point {
        Point::from(LatLng::from_degrees(lat, lon))
    }

    #[test]
    fn triangle() {
        let expected = point_area(
            &s2_point(0.0, 0.0),
            &s2_point(1.0, 0.0),
            &s2_point(0.0, 1.0),
        ) * EARTH_RADIUS_METERS
            * EARTH_RADIUS_METERS;
        let res = area(geography(), "POLYGON ((0 0, 1 0, 0 1, 0 0))");
        assert!(
            ((res - expected) / expected).abs() < 1e-9,
            "{res} vs {expected}"
        );
    }
}
