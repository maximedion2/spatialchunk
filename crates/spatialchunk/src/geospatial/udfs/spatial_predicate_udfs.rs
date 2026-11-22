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

use arrow_array::builder::BooleanBuilder;
use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::cast::as_float64_array;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

use crate::geospatial::geos::joinable_geo::{GeoKernel, JoinableGeo, Planar, Spherical};
use crate::geospatial::geos::raw_geo::GeoType;
use crate::geospatial::join_predicates::st_coveredby::{st_covered_by, st_covers};
use crate::geospatial::join_predicates::st_dwithin::st_dwithin;
use crate::geospatial::join_predicates::st_intersects::st_intersects;
use crate::geospatial::join_predicates::st_touches::st_touches;
use crate::geospatial::join_predicates::st_within::{st_contains, st_within};
use crate::geospatial::udfs::udf_helpers::require_planar_geometry;

// Reads the WKB bytes at `row_idx`, or `None` if the row is null. The signature
// restricts geometry inputs to Binary / BinaryView.
pub(crate) fn wkb_at(array: &ArrayRef, row_idx: usize) -> Option<&[u8]> {
    match array.data_type() {
        DataType::Binary => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("geometry array is Binary");
            (!a.is_null(row_idx)).then(|| a.value(row_idx))
        }
        DataType::BinaryView => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .expect("geometry array is BinaryView");
            (!a.is_null(row_idx)).then(|| a.value(row_idx))
        }
        other => panic!("unsupported geometry array type: {other}"),
    }
}

pub(crate) fn geo_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

// The two-geometry predicate signature: (Binary, Binary) or (BinaryView, BinaryView).
pub(crate) fn binary_pair_signature() -> Signature {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
            TypeSignature::Exact(vec![DataType::BinaryView, DataType::BinaryView]),
        ],
        Volatility::Immutable,
    )
}

// Evaluates a binary geometry predicate row-wise over two WKB columns. A row is
// null if either geometry is null.
fn eval_predicate<K: GeoKernel>(
    args: &ScalarFunctionArgs,
    predicate: impl Fn(&JoinableGeo<K>, &JoinableGeo<K>) -> bool,
) -> Result<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let left = &arrays[0];
    let right = &arrays[1];
    let n = left.len();

    let mut builder = BooleanBuilder::with_capacity(n);
    for i in 0..n {
        match (wkb_at(left, i), wkb_at(right, i)) {
            (Some(a), Some(b)) => {
                let a = JoinableGeo::<K>::from_wkb(a).map_err(geo_err)?;
                let b = JoinableGeo::<K>::from_wkb(b).map_err(geo_err)?;
                builder.append_value(predicate(&a, &b));
            }
            _ => builder.append_null(),
        }
    }
    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

// Both operands have the same type, checked at planning.
fn is_geography(args: &ScalarFunctionArgs) -> Result<bool> {
    let field = args.arg_fields[0].as_ref();
    Ok(GeoType::from_field(field.name(), field)?.is_geography())
}

macro_rules! spatial_predicate_udf {
    ($($struct_name:ident => $udf_name:literal => $func:ident),* $(,)?) => {
        $(
            #[derive(Debug, PartialEq, Eq, Hash)]
            pub struct $struct_name {
                signature: Signature,
            }

            impl Default for $struct_name {
                fn default() -> Self {
                    Self { signature: binary_pair_signature() }
                }
            }

            impl ScalarUDFImpl for $struct_name {
                fn name(&self) -> &str { $udf_name }
                fn signature(&self) -> &Signature {
                    &self.signature
                }
                fn return_type(&self, _: &[DataType]) -> Result<DataType> {
                    Ok(DataType::Boolean)
                }
                fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
                    GeoType::from_fields(
                        self.name(),
                        args.arg_fields[0].as_ref(),
                        args.arg_fields[1].as_ref(),
                    )?;
                    Ok(Arc::new(Field::new(self.name(), DataType::Boolean, true)))
                }
                fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                    if is_geography(&args)? {
                        eval_predicate(&args, $func::<Spherical>)
                    } else {
                        eval_predicate(&args, $func::<Planar>)
                    }
                }
            }
        )*
    }
}

spatial_predicate_udf! {
    StWithinUdf => "st_within" => st_within,
    StContainsUdf => "st_contains" => st_contains,
    StCoveredByUdf => "st_coveredby" => st_covered_by,
    StCoversUdf => "st_covers" => st_covers,
    StIntersectsUdf => "st_intersects" => st_intersects,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StTouchesUdf {
    signature: Signature,
}

impl Default for StTouchesUdf {
    fn default() -> Self {
        Self {
            signature: binary_pair_signature(),
        }
    }
}

impl ScalarUDFImpl for StTouchesUdf {
    fn name(&self) -> &str {
        "st_touches"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        require_planar_geometry(self.name(), args.arg_fields[0].as_ref())?;
        require_planar_geometry(self.name(), args.arg_fields[1].as_ref())?;
        // we already checked for planar geometry, this call is just to validate
        // the crs, a geography would have errored out before getting here.
        GeoType::from_fields(
            self.name(),
            args.arg_fields[0].as_ref(),
            args.arg_fields[1].as_ref(),
        )?;
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, true)))
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        eval_predicate(&args, st_touches)
    }
}

// `st_dwithin(geom, geom, distance)` — like the other predicates, but with a third
// `Float64` distance argument, in meters for geographies, in crs units for
// geometries.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StDWithinUdf {
    signature: Signature,
}

impl Default for StDWithinUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::Binary,
                        DataType::Binary,
                        DataType::Float64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::BinaryView,
                        DataType::BinaryView,
                        DataType::Float64,
                    ]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for StDWithinUdf {
    fn name(&self) -> &str {
        "st_dwithin"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        GeoType::from_fields(
            self.name(),
            args.arg_fields[0].as_ref(),
            args.arg_fields[1].as_ref(),
        )?;
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, true)))
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if is_geography(&args)? {
            eval_dwithin::<Spherical>(&args)
        } else {
            eval_dwithin::<Planar>(&args)
        }
    }
}

fn eval_dwithin<K: GeoKernel>(args: &ScalarFunctionArgs) -> Result<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let left = &arrays[0];
    let right = &arrays[1];
    let distances = as_float64_array(&arrays[2])?;
    let n = left.len();

    let mut builder = BooleanBuilder::with_capacity(n);
    for i in 0..n {
        match (wkb_at(left, i), wkb_at(right, i)) {
            (Some(a), Some(b)) if !distances.is_null(i) => {
                let a = JoinableGeo::<K>::from_wkb(a).map_err(geo_err)?;
                let b = JoinableGeo::<K>::from_wkb(b).map_err(geo_err)?;
                builder.append_value(st_dwithin(&a, &b, distances.value(i)));
            }
            _ => builder.append_null(),
        }
    }
    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}
