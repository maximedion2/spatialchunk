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

//! Helpers shared by the geo udfs: their signatures, and reading a crs
//! argument.

use arrow_schema::DataType;
use datafusion::common::{plan_err, Result, ScalarValue};
use datafusion::logical_expr::{ReturnFieldArgs, Signature, TypeSignature, Volatility};
use serde_json::Value;

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

/// The signature of a udf with an optional trailing crs argument: each of
/// `args` alone, then followed by a string or an integer crs.
pub(crate) fn signature_with_optional_crs(args: Vec<Vec<DataType>>) -> Signature {
    crs_signature(args, true)
}

/// The signature of a udf with a trailing crs argument: each of `args`
/// followed by a string or an integer crs.
pub(crate) fn signature_with_crs(args: Vec<Vec<DataType>>) -> Signature {
    crs_signature(args, false)
}

fn crs_signature(args: Vec<Vec<DataType>>, optional: bool) -> Signature {
    let mut crs_types = vec![Some(DataType::Utf8), Some(DataType::Int64)];
    if optional {
        crs_types.insert(0, None);
    }
    let mut signatures = Vec::new();
    for args in args {
        for crs_type in &crs_types {
            let mut args = args.clone();
            args.extend(crs_type.clone());
            signatures.push(TypeSignature::Exact(args));
        }
    }
    Signature::one_of(signatures, Volatility::Immutable)
}

/// The crs argument at `crs_idx`, if the udf was called with one, e.g. the
/// 'EPSG:4326' of st_point(x, y, 'EPSG:4326'). it ends up on the output field,
/// which is fixed at planning, so it must be a literal.
pub(crate) fn crs_argument(
    udf_name: &str,
    args: &ReturnFieldArgs,
    crs_idx: usize,
) -> Result<Option<Value>> {
    if args.arg_fields.len() <= crs_idx {
        return Ok(None);
    }
    let Some(Some(crs)) = args.scalar_arguments.get(crs_idx) else {
        return plan_err!("{udf_name}: the crs must be a literal, e.g. 'EPSG:4326' or 4326");
    };
    if crs.is_null() {
        return Ok(None);
    }
    match crs {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Ok(Some(Value::String(s.clone()))),
        ScalarValue::Int8(_)
        | ScalarValue::Int16(_)
        | ScalarValue::Int32(_)
        | ScalarValue::Int64(_)
        | ScalarValue::UInt8(_)
        | ScalarValue::UInt16(_)
        | ScalarValue::UInt32(_)
        | ScalarValue::UInt64(_) => Ok(Some(Value::String(crs.to_string()))),
        _ => plan_err!(
            "{udf_name}: the crs must be a string or an integer, got {}",
            crs.data_type()
        ),
    }
}

#[cfg(test)]
mod udf_helpers_tests {
    use std::sync::Arc;

    use arrow_schema::{Field, FieldRef};

    use super::*;

    fn return_field_args<'a>(
        fields: &'a [FieldRef],
        scalars: &'a [Option<&'a ScalarValue>],
    ) -> ReturnFieldArgs<'a> {
        ReturnFieldArgs {
            arg_fields: fields,
            scalar_arguments: scalars,
        }
    }

    #[test]
    fn crs_argument_literals() {
        let fields: Vec<FieldRef> = vec![
            Arc::new(Field::new("g", DataType::Binary, true)),
            Arc::new(Field::new("crs", DataType::Utf8, true)),
        ];
        let crs = |scalar: ScalarValue| {
            crs_argument("f", &return_field_args(&fields, &[None, Some(&scalar)]), 1)
        };

        assert_eq!(
            crs(ScalarValue::from("EPSG:4326")).unwrap(),
            Some(Value::String("EPSG:4326".into()))
        );
        assert_eq!(
            crs(ScalarValue::Int64(Some(4326))).unwrap(),
            Some(Value::String("4326".into()))
        );
        assert_eq!(crs(ScalarValue::Utf8(None)).unwrap(), None);
        assert!(crs(ScalarValue::Float64(Some(1.0))).is_err());

        // not a literal.
        assert!(crs_argument("f", &return_field_args(&fields, &[None, None]), 1).is_err());
        // no crs argument.
        assert_eq!(
            crs_argument("f", &return_field_args(&fields[..1], &[None]), 1).unwrap(),
            None
        );
    }
}
