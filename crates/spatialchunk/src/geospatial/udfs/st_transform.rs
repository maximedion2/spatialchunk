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
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

use super::udf_helpers::{crs_argument, signature_with_crs};
use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};

/// `st_transform(geom, crs)` transforms a geometry from its crs to `crs`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StTransformUdf {
    signature: Signature,
}

impl Default for StTransformUdf {
    fn default() -> Self {
        Self {
            signature: signature_with_crs(vec![vec![DataType::Binary], vec![DataType::BinaryView]]),
        }
    }
}

impl ScalarUDFImpl for StTransformUdf {
    fn name(&self) -> &str {
        "st_transform"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let geo_type = GeoType::from_field(self.name(), args.arg_fields[0].as_ref())?;
        let Some(to_crs) = crs_argument(self.name(), &args, 1)? else {
            return plan_err!("{}: the target crs can't be null", self.name());
        };
        let to = geo_type.transformed(self.name(), to_crs)?;
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
mod transform_tests {
    use arrow::array::AsArray;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::geospatial::test_utils::iou;
    use crate::geospatial::udfs::st_geomfrom::StGeomFromWktUdf;

    // a 100km square in EPSG:5070, and its EPSG:4326 (lon/lat) equivalent
    // from pyproj.
    const SQUARE_5070: &str =
        "POLYGON ((0 2000000, 100000 2000000, 100000 2100000, 0 2100000, 0 2000000))";
    const SQUARE_4326: &str = "POLYGON ((-96 41.00901173, -94.80150187 41.00338024, \
        -94.78619495 41.89674533, -96 41.90245374, -96 41.00901173))";

    const TOL: f64 = 1e-6;

    // runs `expr` on a single row with SQUARE_5070 as column1, and returns the
    // resulting geometry.
    async fn transform_square(expr: &str) -> geos::Geometry<'_> {
        let ctx = SessionContext::new();
        ctx.register_udf(ScalarUDF::from(StGeomFromWktUdf::default()));
        ctx.register_udf(ScalarUDF::from(StTransformUdf::default()));

        let batches = ctx
            .sql(&format!(
                "SELECT {expr} AS g FROM (VALUES ('{SQUARE_5070}'))"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let geoms = batches[0].column(0).as_binary::<i32>();
        geos::Geometry::new_from_wkb(geoms.value(0)).unwrap()
    }

    #[tokio::test]
    async fn transform_matches_pyproj() {
        let res =
            transform_square("st_transform(st_geomfromwkt(column1, 5070), 'EPSG:4326')").await;
        let expected = geos::Geometry::new_from_wkt(SQUARE_4326).unwrap();
        let iou = iou(&res, &expected);
        assert!((iou - 1.0).abs() < TOL, "IoU {iou}");
    }

    #[tokio::test]
    async fn transform_round_trip() {
        let res = transform_square(
            "st_transform(st_transform(st_geomfromwkt(column1, 5070), 4326), 5070)",
        )
        .await;
        let expected = geos::Geometry::new_from_wkt(SQUARE_5070).unwrap();
        let iou = iou(&res, &expected);
        assert!((iou - 1.0).abs() < TOL, "IoU {iou}");
    }
}
