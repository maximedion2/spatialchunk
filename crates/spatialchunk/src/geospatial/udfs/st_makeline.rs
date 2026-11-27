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
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

use super::udf_helpers::binary_pair_signature;
use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};

/// `st_makeline(point, point)` constructs a 2 point line string from two point
/// columns, which must have the same type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StMakeLineUdf {
    signature: Signature,
}

impl Default for StMakeLineUdf {
    fn default() -> Self {
        Self {
            signature: binary_pair_signature(),
        }
    }
}

impl ScalarUDFImpl for StMakeLineUdf {
    fn name(&self) -> &str {
        "st_makeline"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let geo_type = GeoType::from_fields(
            self.name(),
            args.arg_fields[0].as_ref(),
            args.arg_fields[1].as_ref(),
        )?;
        Ok(geo_type.to_field(self.name(), DataType::Binary, true))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let lines = RawGeo::line_from_points(args.arg_fields[0].as_ref(), &arrays[0], &arrays[1])?;
        Ok(lines.into_columnar_value())
    }
}

#[cfg(test)]
mod udf_tests {
    use std::sync::Arc;

    use arrow_array::{Array, BinaryArray};
    use arrow_schema::Field;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, LineStringTrait};
    use serde_json::Value;
    use wkb::reader::Wkb;

    use super::*;
    use crate::geospatial::common::wkb::GeoMetadata;

    fn crs() -> Option<Value> {
        Some(Value::String("EPSG:4326".into()))
    }

    fn point_field(name: &str) -> FieldRef {
        GeoType::geometry(crs()).to_field(name, DataType::Binary, true)
    }

    fn line_coords(bytes: &[u8]) -> Vec<(f64, f64)> {
        let wkb = Wkb::try_new(bytes).unwrap();
        match wkb.as_type() {
            GeometryType::LineString(ls) => ls.coords().map(|c| (c.x(), c.y())).collect(),
            _ => panic!("expected a line string"),
        }
    }

    #[test]
    fn test_udf_names() {
        assert_eq!(StMakeLineUdf::default().name(), "st_makeline");
    }

    #[test]
    fn test_st_makeline_return_field() {
        let fields = [point_field("a"), point_field("b")];
        let args = ReturnFieldArgs {
            arg_fields: &fields,
            scalar_arguments: &[None, None],
        };
        let field = StMakeLineUdf::default()
            .return_field_from_args(args)
            .unwrap();

        assert_eq!(field.data_type(), &DataType::Binary);
        let geo_type = GeoType::from_field(field.name(), field.as_ref()).unwrap();
        assert!(!geo_type.is_geography());
        assert_eq!(GeoMetadata::from_field(&field).unwrap().crs, crs());
    }

    #[test]
    fn test_st_makeline_invoke() {
        let starts = RawGeo::from_points([Some((1.0, 2.0)), None], crs())
            .unwrap()
            .into_array();
        let ends = RawGeo::from_points([Some((3.0, 4.0)), Some((5.0, 6.0))], crs())
            .unwrap()
            .into_array();

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(starts), ColumnarValue::Array(ends)],
            arg_fields: vec![point_field("a"), point_field("b")],
            number_rows: 2,
            return_field: Arc::new(Field::new("out", DataType::Binary, true)),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let out = match StMakeLineUdf::default().invoke_with_args(args).unwrap() {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let out = out.as_any().downcast_ref::<BinaryArray>().unwrap();

        assert_eq!(line_coords(out.value(0)), vec![(1.0, 2.0), (3.0, 4.0)]);
        assert!(out.is_null(1));
    }
}
