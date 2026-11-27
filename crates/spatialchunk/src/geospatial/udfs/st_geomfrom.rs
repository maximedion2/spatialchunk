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

use arrow::compute::cast;
use arrow_array::builder::BinaryBuilder;
use arrow_array::Array;
use arrow_schema::{DataType, FieldRef};
use datafusion::common::cast::as_large_string_array;
use datafusion::common::{exec_datafusion_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};
use geo_types::Geometry;
use wkb::writer::{write_geometry, WriteOptions};
use wkt::TryFromWkt;

use super::udf_helpers::{crs_argument, signature_with_optional_crs};
use crate::geospatial::common::wkb::{geometry_field, GeoMetadata};

/// `st_geomfromwkb(geom [, crs])` stamps an already-WKB binary column as
/// geometry. The bytes are passed through unchanged; only the output field is
/// tagged with the `geoarrow.wkb` extension name, and the crs if given.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StGeomFromWkbUdf {
    signature: Signature,
}

impl Default for StGeomFromWkbUdf {
    fn default() -> Self {
        Self {
            signature: signature_with_optional_crs(vec![
                vec![DataType::Binary],
                vec![DataType::BinaryView],
            ]),
        }
    }
}

impl ScalarUDFImpl for StGeomFromWkbUdf {
    fn name(&self) -> &str {
        "st_geomfromwkb"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    // Tag the output field with the geoarrow.wkb extension name, preserving the
    // input's storage type (Binary or BinaryView).
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let crs = crs_argument(self.name(), &args, 1)?;
        let input = &args.arg_fields[0];
        Ok(geometry_field(
            input.name(),
            input.data_type().clone(),
            input.is_nullable(),
            &GeoMetadata::with_crs(crs),
        ))
    }

    // Identity on the data: the bytes are already WKB.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(args.args.into_iter().next().expect("one argument"))
    }
}

/// `st_geomfromwkt(text [, crs])` parses a WKT string column into WKB, returning
/// a binary column tagged with the `geoarrow.wkb` extension name, and the crs if
/// given.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StGeomFromWktUdf {
    signature: Signature,
}

impl Default for StGeomFromWktUdf {
    fn default() -> Self {
        Self {
            signature: signature_with_optional_crs(vec![
                vec![DataType::Utf8],
                vec![DataType::LargeUtf8],
                vec![DataType::Utf8View],
            ]),
        }
    }
}

impl ScalarUDFImpl for StGeomFromWktUdf {
    fn name(&self) -> &str {
        "st_geomfromwkt"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let crs = crs_argument(self.name(), &args, 1)?;
        Ok(geometry_field(
            args.arg_fields[0].name(),
            DataType::Binary,
            args.arg_fields[0].is_nullable(),
            &GeoMetadata::with_crs(crs),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        // Normalize Utf8 / LargeUtf8 / Utf8View to a single LargeUtf8 accessor.
        // LargeUtf8's i64 offsets can hold any of the three, so the cast never
        // overflows.
        let text = cast(&arrays[0], &DataType::LargeUtf8)?;
        let text = as_large_string_array(&text)?;
        let n = text.len();

        let opts = WriteOptions::default();
        let mut buf = Vec::new();
        let mut builder = BinaryBuilder::with_capacity(n, 0);
        for i in 0..n {
            if text.is_null(i) {
                builder.append_null();
                continue;
            }
            let geom = Geometry::<f64>::try_from_wkt_str(text.value(i))
                .map_err(|e| exec_datafusion_err!("st_geomfromwkt: {e}"))?;
            buf.clear();
            write_geometry(&mut buf, &geom, &opts)
                .map_err(|e| exec_datafusion_err!("st_geomfromwkt: {e}"))?;
            builder.append_value(&buf);
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod geomfrom_tests {
    use arrow_array::{ArrayRef, BinaryArray, StringArray};
    use arrow_schema::Field;
    use datafusion::config::ConfigOptions;
    use geos::Geom;

    use super::*;
    use crate::geospatial::common::constants::GEOARROW_WKB_EXTENSION_NAME;
    use crate::geospatial::test_utils::wkt_to_wkb;

    const POLYS: [&str; 2] = [
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((10 10, 12 10, 12 12, 10 12, 10 10))",
    ];

    // Canonicalizes a WKT string through GEOS so formatting/precision matches the
    // WKT we get back from the UDF output (also produced via GEOS).
    fn norm(wkt: &str) -> String {
        geos::Geometry::new_from_wkt(wkt).unwrap().to_wkt().unwrap()
    }

    fn wkb_to_wkt(bytes: &[u8]) -> String {
        geos::Geometry::new_from_wkb(bytes)
            .unwrap()
            .to_wkt()
            .unwrap()
    }

    fn assert_geometry_field(field: &FieldRef) {
        assert_eq!(
            field
                .metadata()
                .get("ARROW:extension:name")
                .map(String::as_str),
            Some(GEOARROW_WKB_EXTENSION_NAME)
        );
    }

    fn invoke(udf: &dyn ScalarUDFImpl, input: ArrayRef, input_type: DataType) -> BinaryArray {
        let field = Arc::new(Field::new("g", input_type, true));
        let return_field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: std::slice::from_ref(&field),
                scalar_arguments: &[None],
            })
            .unwrap();
        assert_geometry_field(&return_field);

        let n = input.len();
        let out = udf
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Array(input)],
                arg_fields: vec![field],
                number_rows: n,
                return_field,
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap();
        match out {
            ColumnarValue::Array(a) => a.as_any().downcast_ref::<BinaryArray>().unwrap().clone(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn geomfromwkb_stamps_and_passes_through() {
        let wkbs: Vec<Vec<u8>> = POLYS.iter().map(|w| wkt_to_wkb(w)).collect();
        let input = Arc::new(BinaryArray::from_iter_values(
            wkbs.iter().map(|b| b.as_slice()),
        ));

        let out = invoke(&StGeomFromWkbUdf::default(), input, DataType::Binary);

        assert_eq!(wkb_to_wkt(out.value(0)), norm(POLYS[0]));
        assert_eq!(wkb_to_wkt(out.value(1)), norm(POLYS[1]));
    }

    #[test]
    fn geomfromwkt_parses_stamps() {
        let input = Arc::new(StringArray::from_iter_values(POLYS.iter().copied()));

        let out = invoke(&StGeomFromWktUdf::default(), input, DataType::Utf8);

        assert_eq!(wkb_to_wkt(out.value(0)), norm(POLYS[0]));
        assert_eq!(wkb_to_wkt(out.value(1)), norm(POLYS[1]));
    }
}
