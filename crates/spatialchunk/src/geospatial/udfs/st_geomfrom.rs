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

use arrow::compute::cast;
use arrow_array::builder::BinaryBuilder;
use arrow_array::Array;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::cast::as_large_string_array;
use datafusion::common::{exec_datafusion_err, plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use geo_types::Geometry;
use wkb::writer::{write_geometry, WriteOptions};
use wkt::TryFromWkt;

/// The Arrow field-metadata key for an extension type's name.
pub(crate) const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// The GeoArrow extension name that marks a binary column as WKB geometry. A
/// column carrying this on its field metadata is treated as geometry; a bare
/// binary column is not.
pub(crate) const GEOARROW_WKB_EXTENSION_NAME: &str = "geoarrow.wkb";

/// Builds an output `Field` for a geometry column: the given storage type tagged
/// with the `geoarrow.wkb` extension name so downstream operations recognize it.
pub(crate) fn geometry_field(name: &str, storage_type: DataType, nullable: bool) -> FieldRef {
    let mut metadata = HashMap::new();
    metadata.insert(
        ARROW_EXTENSION_NAME_KEY.to_string(),
        GEOARROW_WKB_EXTENSION_NAME.to_string(),
    );
    Arc::new(Field::new(name, storage_type, nullable).with_metadata(metadata))
}

/// Errors unless `field` is tagged as WKB geometry (the `geoarrow.wkb` extension
/// name). Used to reject bare binary columns from geometry UDFs and the spatial
/// join, forcing an explicit `st_geomfromwkb` / `st_geomfromwkt` tag first.
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

/// `st_geomfromwkb(geom)` stamps an already-WKB binary column as geometry. The
/// bytes are passed through unchanged; only the output field is tagged with the
/// `geoarrow.wkb` extension name.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StGeomFromWkbUdf {
    signature: Signature,
}

impl Default for StGeomFromWkbUdf {
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
        let input = &args.arg_fields[0];
        Ok(geometry_field(
            input.name(),
            input.data_type().clone(),
            input.is_nullable(),
        ))
    }

    // Identity on the data: the bytes are already WKB.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(args.args.into_iter().next().expect("one argument"))
    }
}

/// `st_geomfromwkt(text)` parses a WKT string column into WKB, returning a binary
/// column tagged with the `geoarrow.wkb` extension name.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StGeomFromWktUdf {
    signature: Signature,
}

impl Default for StGeomFromWktUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Utf8]),
                    TypeSignature::Exact(vec![DataType::LargeUtf8]),
                    TypeSignature::Exact(vec![DataType::Utf8View]),
                ],
                Volatility::Immutable,
            ),
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
        Ok(geometry_field(
            args.arg_fields[0].name(),
            DataType::Binary,
            args.arg_fields[0].is_nullable(),
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
    use datafusion::config::ConfigOptions;
    use geos::Geom;

    use super::*;
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
