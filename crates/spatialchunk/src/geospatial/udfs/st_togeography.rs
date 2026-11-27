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

use arrow_schema::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};

/// `st_togeography(geom)` transforms a geometry to EPSG:4326 and labels it as a
/// geography (spherical edges). A geography is passed through as is.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StToGeographyUdf {
    signature: Signature,
}

impl Default for StToGeographyUdf {
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

impl ScalarUDFImpl for StToGeographyUdf {
    fn name(&self) -> &str {
        "st_togeography"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let geo_type = GeoType::from_field(self.name(), args.arg_fields[0].as_ref())?;
        let to = geo_type.to_geography(self.name())?;
        Ok(to.to_field(self.name(), DataType::Binary, true))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let to = GeoType::from_field(self.name(), args.return_field.as_ref())?;
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let geos = RawGeo::try_new(args.arg_fields[0].as_ref(), Arc::clone(&arrays[0]))?;
        Ok(geos.transform(to)?.into_columnar_value())
    }
}

#[cfg(test)]
mod togeography_tests {
    use arrow::array::AsArray;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::SessionContext;
    use serde_json::Value;

    use super::*;
    use crate::geospatial::common::constants::GEOGRAPHY_CRS;
    use crate::geospatial::common::wkb::GeoMetadata;
    use crate::geospatial::test_utils::iou;
    use crate::geospatial::udfs::st_geomfrom::StGeomFromWktUdf;

    // a 100km square in EPSG:5070, and its EPSG:4326 (lon/lat) equivalent
    // from pyproj.
    const SQUARE_5070: &str =
        "POLYGON ((0 2000000, 100000 2000000, 100000 2100000, 0 2100000, 0 2000000))";
    const SQUARE_4326: &str = "POLYGON ((-96 41.00901173, -94.80150187 41.00338024, \
        -94.78619495 41.89674533, -96 41.90245374, -96 41.00901173))";

    const TOL: f64 = 1e-6;

    #[tokio::test]
    async fn togeography_transforms_and_labels() {
        let ctx = SessionContext::new();
        ctx.register_udf(ScalarUDF::from(StGeomFromWktUdf::default()));
        ctx.register_udf(ScalarUDF::from(StToGeographyUdf::default()));

        let batches = ctx
            .sql(&format!(
                "SELECT st_togeography(st_geomfromwkt(column1, 5070)) AS g \
                 FROM (VALUES ('{SQUARE_5070}'))"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let metadata = GeoMetadata::from_field(batches[0].schema().field(0)).unwrap();
        assert!(metadata.is_geography());
        assert_eq!(metadata.crs, Some(Value::String(GEOGRAPHY_CRS.into())));

        let geoms = batches[0].column(0).as_binary::<i32>();
        let res = geos::Geometry::new_from_wkb(geoms.value(0)).unwrap();
        let expected = geos::Geometry::new_from_wkt(SQUARE_4326).unwrap();
        let iou = iou(&res, &expected);
        assert!((iou - 1.0).abs() < TOL, "IoU {iou}");
    }
}
