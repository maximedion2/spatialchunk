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

use arrow_schema::{DataType, FieldRef};
use datafusion::common::cast::as_float64_array;
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

use super::udf_helpers::{crs_argument, signature_with_optional_crs, GeoMetadata};
use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};

/// `st_point(x, y [, crs])` constructs a WKB point from two `Float64` coordinate
/// columns, returning a binary array, tagged with the crs if given.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StPointUdf {
    signature: Signature,
}

impl Default for StPointUdf {
    fn default() -> Self {
        Self {
            // Exact Float64 args; DataFusion coerces other numeric inputs to Float64.
            signature: signature_with_optional_crs(vec![vec![
                DataType::Float64,
                DataType::Float64,
            ]]),
        }
    }
}

impl ScalarUDFImpl for StPointUdf {
    fn name(&self) -> &str {
        "st_point"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    // st_point builds a geometry, so its output is tagged as WKB geometry.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let crs = crs_argument(self.name(), &args, 2)?;
        Ok(GeoType::geometry(crs).to_field(self.name(), DataType::Binary, true))
    }

    // the crs was resolved at planning, it's on the return field.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let xs = as_float64_array(&arrays[0])?;
        let ys = as_float64_array(&arrays[1])?;
        let crs = GeoMetadata::from_field(&args.return_field)?.crs;

        let points = xs.iter().zip(ys.iter()).map(|(x, y)| x.zip(y));
        let points = RawGeo::from_points(points, crs)?;
        Ok(points.into_columnar_value())
    }
}

#[cfg(test)]
mod udf_tests {
    use std::sync::Arc;

    use arrow_array::{Array, BinaryArray, Float64Array};
    use arrow_schema::Field;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
    use wkb::reader::Wkb;

    use super::*;

    #[test]
    fn test_udf_names() {
        assert_eq!(StPointUdf::default().name(), "st_point");
    }

    fn point_coords(bytes: &[u8]) -> (f64, f64) {
        let wkb = Wkb::try_new(bytes).unwrap();
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let c = p.coord().unwrap();
                (c.x(), c.y())
            }
            _ => panic!("expected a point"),
        }
    }

    #[test]
    fn test_st_point_invoke() {
        let xs = Float64Array::from(vec![Some(1.0), Some(-3.5), None]);
        let ys = Float64Array::from(vec![Some(2.0), Some(4.25), Some(9.0)]);
        let n = xs.len();

        let udf = StPointUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(xs)),
                ColumnarValue::Array(Arc::new(ys)),
            ],
            arg_fields: vec![
                Arc::new(Field::new("x", DataType::Float64, true)),
                Arc::new(Field::new("y", DataType::Float64, true)),
            ],
            number_rows: n,
            return_field: Arc::new(Field::new("out", DataType::Binary, true)),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let out = match udf.invoke_with_args(args).unwrap() {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let out = out.as_any().downcast_ref::<BinaryArray>().unwrap();

        assert_eq!(point_coords(out.value(0)), (1.0, 2.0));
        assert_eq!(point_coords(out.value(1)), (-3.5, 4.25));
        assert!(out.is_null(2));
    }
}
