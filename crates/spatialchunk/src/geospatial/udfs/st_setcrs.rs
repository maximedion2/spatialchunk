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

use super::udf_helpers::{
    crs_argument, geometry_field, require_planar_geometry, signature_with_crs, GeoMetadata,
};

/// `st_setcrs(geom, crs)` sets the crs of a geometry, replacing any existing
/// one. The coordinates are not transformed, see `st_transform` for that.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StSetCrsUdf {
    signature: Signature,
}

impl Default for StSetCrsUdf {
    fn default() -> Self {
        Self {
            signature: signature_with_crs(vec![vec![DataType::Binary], vec![DataType::BinaryView]]),
        }
    }
}

impl ScalarUDFImpl for StSetCrsUdf {
    fn name(&self) -> &str {
        "st_setcrs"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = &args.arg_fields[0];
        require_planar_geometry(self.name(), input)?;
        let crs = crs_argument(self.name(), &args, 1)?;
        Ok(geometry_field(
            input.name(),
            input.data_type().clone(),
            input.is_nullable(),
            &GeoMetadata::with_crs(crs),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(args.args.into_iter().next().expect("two arguments"))
    }
}

#[cfg(test)]
mod setcrs_tests {
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::SessionContext;
    use serde_json::Value;

    use super::*;
    use crate::geospatial::udfs::st_geomfrom::StGeomFromWktUdf;

    #[tokio::test]
    async fn setcrs_sets_the_crs_on_the_output_field() {
        let ctx = SessionContext::new();
        ctx.register_udf(ScalarUDF::from(StGeomFromWktUdf::default()));
        ctx.register_udf(ScalarUDF::from(StSetCrsUdf::default()));

        let batches = ctx
            .sql(
                "SELECT st_setcrs(st_geomfromwkt(column1), 'EPSG:4326') AS g_str, \
                        st_setcrs(st_geomfromwkt(column1), 4326) AS g_int \
                 FROM (VALUES ('POINT (1 2)'), ('POINT (3 4)'))",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let schema = batches[0].schema();
        let crs = |name: &str| {
            GeoMetadata::from_field(schema.field_with_name(name).unwrap())
                .unwrap()
                .crs
        };
        assert_eq!(crs("g_str"), Some(Value::String("EPSG:4326".into())));
        assert_eq!(crs("g_int"), Some(Value::String("4326".into())));
    }
}
