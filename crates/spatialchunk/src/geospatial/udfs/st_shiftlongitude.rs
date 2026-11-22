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

use arrow_array::Float64Array;
use arrow_schema::{DataType, FieldRef};
use datafusion::common::cast::as_float64_array;
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

use crate::geospatial::geos::raw_geo::{shift_longitude, GeoType, RawGeo};

/// `st_shiftlongitude(geom)` wraps a geometry's longitudes to -180..180.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StShiftLongitudeUdf {
    signature: Signature,
}

impl Default for StShiftLongitudeUdf {
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

impl ScalarUDFImpl for StShiftLongitudeUdf {
    fn name(&self) -> &str {
        "st_shiftlongitude"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Binary)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let geo_type = GeoType::from_field(self.name(), args.arg_fields[0].as_ref())?;
        if geo_type.is_geography() {
            return plan_err!("{}: expected a geometry, got a geography", self.name());
        }
        Ok(geo_type.to_field(self.name(), DataType::Binary, true))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let geos = RawGeo::try_new(args.arg_fields[0].as_ref(), Arc::clone(&arrays[0]))?;
        Ok(geos.shift_longitude()?.into_columnar_value())
    }
}

/// `shiftlongitude(x)` wraps a longitude column to -180..180, like
/// st_shiftlongitude does for a geometry's vertices.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ShiftLongitudeUdf {
    signature: Signature,
}

impl Default for ShiftLongitudeUdf {
    fn default() -> Self {
        Self {
            // DataFusion coerces other numeric inputs to Float64.
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ShiftLongitudeUdf {
    fn name(&self) -> &str {
        "shiftlongitude"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let xs = as_float64_array(&arrays[0])?;
        let out: Float64Array = xs.unary(shift_longitude);
        Ok(ColumnarValue::Array(Arc::new(out)))
    }
}
