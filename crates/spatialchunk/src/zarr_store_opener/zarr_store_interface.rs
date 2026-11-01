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

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::*;
use arrow::compute::kernels::aggregate::{max as arrow_max, min as arrow_min};
use arrow::compute::kernels::cmp::{eq, gt, lt};
use arrow::compute::{cast, cast_with_options, concat, nullif, or, take, CastOptions};
use arrow::datatypes::*;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use base64::prelude::{Engine, BASE64_STANDARD};
use bytes::Bytes;
use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion::physical_plan::PhysicalExpr;
use futures::future::{BoxFuture, Shared};
use serde_json::{Map, Value};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinSet;
use zarrs::array::codec::api::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
    Int8DataType, NumpyDateTime64DataType, NumpyTimeDelta64DataType, StringDataType,
    UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::{
    Array, ArrayBytes, ArraySubset, ChunkShapeTraits, DataType as zDataType, ElementOwned,
};
use zarrs::metadata_ext::data_type::NumpyTimeUnit;
use zarrs_storage::AsyncReadableListableStorageTraits;

use super::filter::ZarrChunkFilter;
use super::io_runtime::IoRuntime;
use super::metrics::ZarrMetrics;
use super::util::{ColumnNames, NumElements};
use super::zarr_errors::{ZarrQueryError, ZarrQueryResult};

//***************************************************
// A helper struct that handles the special case where an
// array is a 1D coordinate (that can be broadcasted)
//***************************************************
struct BroadcastableCoord {
    field: FieldRef,
    values: ArrayRef,

    // the position of the coordinate in the full dimensionality.
    position: usize,
}

impl BroadcastableCoord {
    // the range of the coordinate's dimension, out of one optional range per
    // dimension of the full dimensionality.
    fn range<'a>(&self, ranges: &'a [Option<Range<u64>>]) -> ZarrQueryResult<&'a Range<u64>> {
        let name = self.field.name();
        let range = ranges
            .get(self.position)
            .ok_or_else(|| {
                ZarrQueryError::InvalidCompute(format!(
                    "Coordinate {name} is outside of the broadcast dimensions"
                ))
            })?
            .as_ref()
            .ok_or_else(|| {
                ZarrQueryError::InvalidCompute(format!(
                    "No range provided for the dimension of coordinate {name}"
                ))
            })?;
        if range.end > self.values.len() as u64 {
            return Err(ZarrQueryError::InvalidCompute(format!(
                "Range {range:?} is out of bounds for coordinate {name}"
            )));
        }
        Ok(range)
    }

    // the coordinate values over the range of its dimension, not broadcast.
    fn slice(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<ArrayRef> {
        let range = self.range(ranges)?;
        Ok(self.values.slice(
            range.start as usize,
            range.end.saturating_sub(range.start) as usize,
        ))
    }

    // the coordinate values broadcast over the dimensions with a range, see
    // the `Broadcastable` trait.
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<ArrayRef> {
        let sliced = self.slice(ranges)?;

        let lens: Vec<u64> = ranges
            .iter()
            .flatten()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        if lens.num_elements() == 0 {
            return Ok(new_empty_array(self.field.data_type()));
        }

        // the output is each value of the range repeated `stride` times (the
        // axes after the coordinate's), with that block tiled `outer` times
        // (the axes before it).
        let axis = ranges[..self.position].iter().flatten().count();
        let len = lens[axis];
        let stride = lens[axis + 1..].num_elements();
        let outer = lens[..axis].num_elements();

        let block = if stride == 1 {
            sliced
        } else {
            let indices: UInt64Array = (0..len * stride).map(|i| i / stride).collect();
            take(&sliced, &indices, None)?
        };
        if outer == 1 {
            Ok(block)
        } else {
            Ok(concat(&vec![block.as_ref(); outer as usize])?)
        }
    }
}

trait Broadcastable {
    // takes one optional range per dimension of the full dimensionality.
    // the output spans the dimensions with a range, in C order. dimensions
    // without a coordinate just repeat the values along that axis.
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch>;

    // takes the ranges of a chunk along every dimension, and broadcasts only
    // over the dimensions of the coordinates.
    fn broadcast_chunk(&self, chunk_ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch>;
}

impl Broadcastable for [&BroadcastableCoord] {
    fn broadcast(&self, ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch> {
        let num_rows: u64 = ranges
            .iter()
            .flatten()
            .map(|r| r.end.saturating_sub(r.start))
            .product();

        let mut fields = Vec::with_capacity(self.len());
        let mut columns = Vec::with_capacity(self.len());
        for coord in self {
            fields.push(coord.field.clone());
            columns.push(coord.broadcast(ranges)?);
        }

        Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows as usize)),
        )?)
    }

    fn broadcast_chunk(&self, chunk_ranges: &[Option<Range<u64>>]) -> ZarrQueryResult<RecordBatch> {
        let mut ranges = vec![None; chunk_ranges.len()];
        for coord in self {
            ranges[coord.position] = chunk_ranges[coord.position].clone();
        }
        self.broadcast(&ranges)
    }
}

//***************************************************
// A helper struct for the geometry of the chunk grid.
//***************************************************
#[derive(Clone, Default)]
struct ChunkGrid {
    array_shape: Vec<u64>,
    grid_shape: Vec<u64>,

    // the chunk shape in the metadata, the chunks at the array edges can be
    // smaller.
    chunk_shape: Vec<u64>,
}

impl ChunkGrid {
    // technically getting the chunk shape requires a chunk index, but zarrs
    // doesn't return a chunk size that depends on the index for regular
    // grids, so we retrieve "chunk 0" and adjust for array edges when reading.
    // so only regular grids are supported here, for now.
    fn from_array<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Self> {
        Ok(Self {
            array_shape: arr.shape().to_vec(),
            chunk_shape: arr
                .chunk_shape(&vec![0; arr.dimensionality()])?
                .to_array_shape(),
            grid_shape: arr.chunk_grid_shape().to_vec(),
        })
    }

    // appends the dimensions of another grid after this grid's dimensions.
    fn extend(&mut self, other: ChunkGrid) {
        self.array_shape.extend(other.array_shape);
        self.chunk_shape.extend(other.chunk_shape);
        self.grid_shape.extend(other.grid_shape);
    }

    // the real chunk shape for a given index, adjusted for the array edges.
    fn chunk_shape(&self, chk_idx: &[u64]) -> Vec<u64> {
        chk_idx
            .iter()
            .zip(self.array_shape.iter())
            .zip(self.chunk_shape.iter())
            .map(|((i, a), c)| std::cmp::min(a - i * c, *c))
            .collect()
    }

    // the range of values the chunk covers along every dimension.
    fn ranges(&self, chk_idx: &[u64]) -> Vec<Option<Range<u64>>> {
        chk_idx
            .iter()
            .zip(self.chunk_shape.iter())
            .zip(self.chunk_shape(chk_idx))
            .map(|((i, c), real_c)| Some(i * c..i * c + real_c))
            .collect()
    }

    // whether the chunk is at the edge of the array along any dimension.
    fn is_edge(&self, chk_idx: &[u64]) -> bool {
        chk_idx
            .iter()
            .zip(self.grid_shape.iter())
            .any(|(i, g)| *i == g - 1)
    }
}

impl FromIterator<ChunkGrid> for ChunkGrid {
    fn from_iter<I: IntoIterator<Item = ChunkGrid>>(iter: I) -> Self {
        let mut grid = ChunkGrid::default();
        for other in iter {
            grid.extend(other);
        }
        grid
    }
}

//***************************************************
// A helper struct to resolve data arrays and 1D coordinates
// and validate everything.
//***************************************************
struct DataArraysMeta {
    dim_names: Option<Vec<String>>,
    grid: ChunkGrid,
}

impl DataArraysMeta {
    fn new<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Self> {
        Ok(Self {
            dim_names: dim_names(arr)?,
            grid: ChunkGrid::from_array(arr)?,
        })
    }

    fn validate<T: ?Sized>(&self, name: &str, arr: &Array<T>) -> ZarrQueryResult<()> {
        let other = Self::new(arr)?;
        if other.grid.array_shape != self.grid.array_shape {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Shape of array {name} doesn't match the shape of the other data arrays"
            )));
        }
        if other.grid.chunk_shape != self.grid.chunk_shape {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Chunk shape of array {name} doesn't match the chunk shape of the other data arrays"
            )));
        }
        if other.dim_names != self.dim_names {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Dimension names of array {name} don't match the dimension names of the other data arrays"
            )));
        }
        Ok(())
    }

    // returns the position of the coordinate in the data dimensionality.
    // also validate that the coord exists and has the right size.
    fn validate_coord(&self, name: &str, values: &ArrayRef) -> ZarrQueryResult<usize> {
        let dim_names = self
            .dim_names
            .as_ref()
            .ok_or(ZarrQueryError::InvalidMetadata(
                "Dimension names are required on data arrays when there are coordinates".into(),
            ))?;
        let position = dim_names.iter().position(|d| d == name).ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!(
                "Coordinate {name} is not a dimension of the data arrays"
            ))
        })?;
        if values.len() as u64 != self.grid.array_shape[position] {
            return Err(ZarrQueryError::InvalidMetadata(format!(
                "Length of coordinate {name} doesn't match the data size along its dimension"
            )));
        }
        Ok(position)
    }
}

// when only coordinates are requested, each one is its own dimension, in the
// order they were requested, with its own chunking.
fn coords_only_meta(
    coords: Vec<(String, FieldRef, ArrayRef, ChunkGrid)>,
) -> (HashMap<String, BroadcastableCoord>, ChunkGrid) {
    let (coords, grids): (Vec<_>, Vec<_>) = coords
        .into_iter()
        .map(|(name, field, values, grid)| ((name, field, values), grid))
        .unzip();

    let broadcastable_coords = coords
        .into_iter()
        .enumerate()
        .map(|(position, (name, field, values))| {
            (
                name,
                BroadcastableCoord {
                    position,
                    field,
                    values,
                },
            )
        })
        .collect();
    (broadcastable_coords, grids.into_iter().collect())
}

//***************************************************
// The time conversions that can be expressed with CF attributes,
// for int and float arrays. the output is always in nanoseconds.
//***************************************************
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfTimeUnit {
    Days,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl CfTimeUnit {
    // the lower case names, singular or plural, the same list xarray supports.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "days" | "day" => Some(Self::Days),
            "hours" | "hour" => Some(Self::Hours),
            "minutes" | "minute" => Some(Self::Minutes),
            "seconds" | "second" => Some(Self::Seconds),
            "milliseconds" | "millisecond" => Some(Self::Milliseconds),
            "microseconds" | "microsecond" => Some(Self::Microseconds),
            "nanoseconds" | "nanosecond" => Some(Self::Nanoseconds),
            _ => None,
        }
    }

    fn nanos(&self) -> i64 {
        match self {
            Self::Days => 86_400_000_000_000,
            Self::Hours => 3_600_000_000_000,
            Self::Minutes => 60_000_000_000,
            Self::Seconds => 1_000_000_000,
            Self::Milliseconds => 1_000_000,
            Self::Microseconds => 1_000,
            Self::Nanoseconds => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfCalendar {
    Standard,
    ProlepticGregorian,
}

impl CfCalendar {
    // a missing calendar means the standard one. the other calendars have
    // dates that arrow timestamps can't represent (e.g. feb 30 in 360_day).
    fn from_name(name: Option<&str>) -> ZarrQueryResult<Self> {
        match name.map(|n| n.trim().to_lowercase()).as_deref() {
            None | Some("standard") | Some("gregorian") => Ok(Self::Standard),
            Some("proleptic_gregorian") => Ok(Self::ProlepticGregorian),
            Some(other) => Err(ZarrQueryError::InvalidMetadata(format!(
                "Unsupported calendar {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TimeConversion {
    // e.g. units "hours", with a "timedelta64" dtype attribute.
    Duration {
        unit: CfTimeUnit,
    },
    // e.g. units "days since 1970-01-01", the reference is in nanoseconds
    // since the unix epoch.
    TimeSince {
        unit: CfTimeUnit,
        reference_ns: i64,
        calendar: CfCalendar,
    },
}

impl TimeConversion {
    // the caller decides if the array is a duration, xarray only decodes
    // durations when the dtype attribute marks them, so units like "hours"
    // alone are left as raw values.
    fn duration(units: &str) -> ZarrQueryResult<Self> {
        let unit = CfTimeUnit::from_name(units.trim()).ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!("Unsupported duration units {units}"))
        })?;
        Ok(Self::Duration { unit })
    }

    fn time_since(units: &str, calendar: Option<&str>) -> ZarrQueryResult<Self> {
        let lower = units.trim().to_lowercase();
        let (unit, reference) = lower.split_once(" since ").ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!(
                "Expected <unit> since <datetime>, got {units}"
            ))
        })?;

        let unit = CfTimeUnit::from_name(unit.trim()).ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!("Unsupported time units {units}"))
        })?;

        let calendar = CfCalendar::from_name(calendar)?;
        let reference = parse_reference_datetime(reference.trim()).ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!("Unsupported reference datetime in {units}"))
        })?;

        // the outputs are nanosecond timestamps, so the reference has to fit
        // in that range too. this also rules out dates before 1582, where the
        // standard calendar would follow julian rules.
        let reference_ns = reference.and_utc().timestamp_nanos_opt().ok_or_else(|| {
            ZarrQueryError::InvalidMetadata(format!(
                "Reference datetime in {units} is out of the nanosecond timestamp range"
            ))
        })?;

        Ok(Self::TimeSince {
            unit,
            reference_ns,
            calendar,
        })
    }

    fn data_type(&self) -> DataType {
        match self {
            Self::Duration { .. } => DataType::Duration(TimeUnit::Nanosecond),
            Self::TimeSince { .. } => DataType::Timestamp(TimeUnit::Nanosecond, None),
        }
    }

    // converts int or float values to nanoseconds. integers are converted
    // exactly, floats are rounded to the nearest nanosecond and NaN becomes
    // null. values out of the nanosecond range are an error.
    fn convert(&self, arr: ArrayRef) -> ZarrQueryResult<ArrayRef> {
        let (unit, offset_ns) = match self {
            Self::Duration { unit } => (unit, 0),
            Self::TimeSince {
                unit, reference_ns, ..
            } => (unit, *reference_ns),
        };
        let nanos = unit.nanos();
        let out_of_range = || {
            ZarrQueryError::InvalidCompute(
                "Time value is out of the nanosecond timestamp range".into(),
            )
        };

        let values: Vec<Option<i64>> = if arr.data_type().is_floating() {
            let floats = cast(&arr, &DataType::Float64)?;
            floats
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| match v {
                    Some(v) if !v.is_nan() => {
                        let delta = (v * nanos as f64).round();
                        // i64::MAX as f64 is 2^63, which doesn't fit, hence the <.
                        if !(delta >= i64::MIN as f64 && delta < i64::MAX as f64) {
                            return Err(out_of_range());
                        }
                        (delta as i64)
                            .checked_add(offset_ns)
                            .map(Some)
                            .ok_or_else(out_of_range)
                    }
                    _ => Ok(None),
                })
                .collect::<ZarrQueryResult<_>>()?
        } else {
            // not a safe cast, so that u64 values past i64::MAX are an error
            // rather than null.
            let options = CastOptions {
                safe: false,
                ..Default::default()
            };
            let ints = cast_with_options(&arr, &DataType::Int64, &options)?;
            ints.as_primitive::<Int64Type>()
                .iter()
                .map(|v| {
                    v.map(|v| {
                        v.checked_mul(nanos)
                            .and_then(|v| v.checked_add(offset_ns))
                            .ok_or_else(out_of_range)
                    })
                    .transpose()
                })
                .collect::<ZarrQueryResult<_>>()?
        };

        Ok(match self {
            Self::Duration { .. } => Arc::new(DurationNanosecondArray::from(values)) as ArrayRef,
            Self::TimeSince { .. } => Arc::new(TimestampNanosecondArray::from(values)),
        })
    }
}

// parses the udunits style reference datetime, in utc. the fields don't need
// to be padded, the time and the offset are optional, e.g. "1-1-1",
// "1970-01-01T00:00:00Z", "2000-1-1 12:30:00.5 +5:30". the input is lower case.
fn parse_reference_datetime(s: &str) -> Option<NaiveDateTime> {
    let (date, rest) = match s
        .char_indices()
        .find(|(_, c)| *c == 't' || c.is_whitespace())
    {
        Some((i, c)) => (&s[..i], s[i + c.len_utf8()..].trim_start()),
        None => (s, ""),
    };

    let mut parts = date.split('-');
    let year = parse_digits(parts.next()?)?;
    let month = parse_digits(parts.next()?)?;
    let day = parse_digits(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }

    // the time is everything up to the offset, if any.
    let time_len = rest
        .find(|c: char| !(c.is_ascii_digit() || c == ':' || c == '.'))
        .unwrap_or(rest.len());
    let (time, offset) = (&rest[..time_len], rest[time_len..].trim());
    let (hour, min, sec, nanos) = match time.is_empty() {
        true => (0, 0, 0, 0),
        false => parse_time(time)?,
    };

    let local =
        NaiveDate::from_ymd_opt(year, month, day)?.and_hms_nano_opt(hour, min, sec, nanos)?;
    local.checked_sub_signed(TimeDelta::seconds(parse_offset_secs(offset)?))
}

// "h:m", "h:m:s" or "h:m:s.f", fractions past nanoseconds are truncated.
fn parse_time(time: &str) -> Option<(u32, u32, u32, u32)> {
    let mut parts = time.split(':');
    let hour = parse_digits(parts.next()?)?;
    let min = parse_digits(parts.next()?)?;
    let (sec, nanos) = match parts.next() {
        None => (0, 0),
        Some(s) => {
            let (whole, frac) = s.split_once('.').unwrap_or((s, ""));
            let nanos = match frac.is_empty() {
                true => 0,
                false => {
                    let frac: String = frac.chars().take(9).collect();
                    parse_digits(&format!("{frac:0<9}"))?
                }
            };
            (parse_digits(whole)?, nanos)
        }
    };
    if parts.next().is_some() {
        return None;
    }
    Some((hour, min, sec, nanos))
}

// "", "z", "utc", or a signed offset like "+5", "-6:00", "+0530".
fn parse_offset_secs(offset: &str) -> Option<i64> {
    if matches!(offset, "" | "z" | "utc") {
        return Some(0);
    }
    let (sign, rest) = match offset.as_bytes().first()? {
        b'+' => (1, &offset[1..]),
        b'-' => (-1, &offset[1..]),
        _ => return None,
    };
    let (hours, mins): (i64, i64) = match rest.split_once(':') {
        Some((h, m)) => (parse_digits(h)?, parse_digits(m)?),
        None if rest.len() == 4 => (parse_digits(&rest[..2])?, parse_digits(&rest[2..])?),
        None => (parse_digits(rest)?, 0),
    };
    if hours > 23 || mins > 59 {
        return None;
    }
    Some(sign * (hours * 3600 + mins * 60))
}

// a non empty, unsigned run of digits.
fn parse_digits<N: std::str::FromStr>(s: &str) -> Option<N> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

//***************************************************
// Sets the values that represent missing data to null, from the
// _FillValue, missing_value and valid range attributes. the mask is
// applied before unpacking (i.e. scale and offset).
//***************************************************
struct DataMask {
    // _FillValue and missing_value. NaN is a separate flag since it never
    // compares equal.
    values: Option<Vec<Scalar<ArrayRef>>>,
    nan_is_null: bool,

    // valid_min and valid_max, or valid_range, the bounds are inclusive.
    min: Option<Scalar<ArrayRef>>,
    max: Option<Scalar<ArrayRef>>,
}

impl DataMask {
    // None if none of the attributes are present.
    fn new(
        attrs: &Map<String, Value>,
        raw_type: &DataType,
        reinterpret_type: Option<&DataType>,
    ) -> Result<Option<Self>, String> {
        let mut fill_json = Vec::new();
        if let Some(v) = attrs.get("_FillValue") {
            fill_json.push(v);
        }
        match attrs.get("missing_value") {
            Some(Value::Array(values)) => fill_json.extend(values),
            Some(v) => fill_json.push(v),
            None => {}
        }
        let (values, nan_is_null) = match fill_json.is_empty() {
            true => (None, false),
            false => {
                let (values, nan) = parse_scalars(&fill_json, raw_type, reinterpret_type)?;
                (Some(values).filter(|v| !v.is_empty()), nan)
            }
        };

        // a bound is a single value, as a number or a single element list.
        let bound = |key: &str| -> Result<Option<Scalar<ArrayRef>>, String> {
            let value = match attrs.get(key) {
                None => return Ok(None),
                Some(Value::Array(values)) if values.len() == 1 => &values[0],
                Some(v) => v,
            };
            let (mut scalars, nan) = parse_scalars(&[value], raw_type, reinterpret_type)?;
            match (scalars.pop(), nan) {
                (Some(scalar), false) => Ok(Some(scalar)),
                _ => Err(format!("invalid {key} {value}")),
            }
        };
        let (min, max) = match attrs.get("valid_range") {
            None => (bound("valid_min")?, bound("valid_max")?),
            Some(_) if attrs.contains_key("valid_min") || attrs.contains_key("valid_max") => {
                return Err("valid_range can't be combined with valid_min or valid_max".into())
            }
            Some(Value::Array(range)) if range.len() == 2 => {
                let (mut scalars, nan) =
                    parse_scalars(&[&range[0], &range[1]], raw_type, reinterpret_type)?;
                if nan || scalars.len() != 2 {
                    return Err(format!("invalid valid_range {range:?}"));
                }
                let max = scalars.pop();
                (scalars.pop(), max)
            }
            Some(other) => return Err(format!("invalid valid_range {other}")),
        };

        if values.is_none() && !nan_is_null && min.is_none() && max.is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            values,
            nan_is_null,
            min,
            max,
        }))
    }

    // the array is in the packed type.
    fn apply(&self, arr: ArrayRef) -> ZarrQueryResult<ArrayRef> {
        // true where the row becomes null.
        let mut nulls = match self.nan_is_null {
            true => is_nan(&arr)?,
            false => BooleanArray::from(vec![false; arr.len()]),
        };
        for value in self.values.iter().flatten() {
            nulls = or(&nulls, &eq(&arr, value)?)?;
        }
        if let Some(min) = &self.min {
            nulls = or(&nulls, &lt(&arr, min)?)?;
        }
        if let Some(max) = &self.max {
            nulls = or(&nulls, &gt(&arr, max)?)?;
        }
        Ok(nullif(&arr, &nulls)?)
    }
}

fn is_nan(arr: &ArrayRef) -> ZarrQueryResult<BooleanArray> {
    match arr.data_type() {
        DataType::Float32 => Ok(BooleanArray::from_unary(
            arr.as_primitive::<Float32Type>(),
            |v| v.is_nan(),
        )),
        DataType::Float64 => Ok(BooleanArray::from_unary(
            arr.as_primitive::<Float64Type>(),
            |v| v.is_nan(),
        )),
        other => Err(ZarrQueryError::InvalidCompute(format!(
            "NaN mask on a non float array of type {other}"
        ))),
    }
}

// the integer type of the same width with the other signedness, for
// _Unsigned "true" on signed integers or "false" on unsigned integers.
// otherwise there's nothing to reinterpret.
fn reinterpret_type(raw_type: &DataType, value: &Value) -> Result<Option<DataType>, String> {
    let unsigned = match value {
        Value::Bool(b) => *b,
        Value::String(s) if s.eq_ignore_ascii_case("true") => true,
        Value::String(s) if s.eq_ignore_ascii_case("false") => false,
        other => return Err(format!("invalid _Unsigned value {other}")),
    };
    if !raw_type.is_integer() {
        return Err(format!(
            "_Unsigned is only supported on int arrays, not {raw_type}"
        ));
    }
    Ok(match (unsigned, raw_type) {
        (true, DataType::Int8) => Some(DataType::UInt8),
        (true, DataType::Int16) => Some(DataType::UInt16),
        (true, DataType::Int32) => Some(DataType::UInt32),
        (true, DataType::Int64) => Some(DataType::UInt64),
        (false, DataType::UInt8) => Some(DataType::Int8),
        (false, DataType::UInt16) => Some(DataType::Int16),
        (false, DataType::UInt32) => Some(DataType::Int32),
        (false, DataType::UInt64) => Some(DataType::Int64),
        _ => None,
    })
}

// reinterprets the bits of an integer array as another integer type of the
// same width.
fn reinterpret(arr: ArrayRef, data_type: &DataType) -> ZarrQueryResult<ArrayRef> {
    let data = arr
        .to_data()
        .into_builder()
        .data_type(data_type.clone())
        .build()?;
    Ok(make_array(data))
}

// the values as scalars of the reinterpreted type (or the raw type when
// there's no reinterpretation), and whether one of them is NaN, which is
// left out of the scalars. the values are in the raw type, values that don't
// fit it are an error.
fn parse_scalars(
    values: &[&Value],
    raw_type: &DataType,
    reinterpret_type: Option<&DataType>,
) -> Result<(Vec<Scalar<ArrayRef>>, bool), String> {
    let to_err = |e: ZarrQueryError| e.to_string();

    // the values are cast as one array, which validates them all at once,
    // then split into scalars.
    let to_scalars = |arr: ArrayRef| -> Vec<Scalar<ArrayRef>> {
        (0..arr.len())
            .map(|i| Scalar::new(arr.slice(i, 1)))
            .collect()
    };

    // _Unsigned only applies to integers, so floats are never reinterpreted.
    if raw_type.is_floating() {
        let mut floats = Vec::with_capacity(values.len());
        let mut nan = false;
        for v in values {
            let f = json_to_f64(v).ok_or_else(|| format!("invalid value {v}"))?;
            match f.is_nan() {
                true => nan = true,
                false => floats.push(f),
            }
        }
        let arr: ArrayRef = Arc::new(Float64Array::from(floats));
        let arr = cast(&arr, raw_type).map_err(|e| to_err(e.into()))?;
        return Ok((to_scalars(arr), nan));
    }

    if !raw_type.is_integer() {
        return Err(format!(
            "fill values and valid ranges are not supported on {raw_type} arrays"
        ));
    }

    // the values are written in the raw type, like the data, and
    // reinterpreted the same way.
    let mut arr = int_array(values, raw_type)?;
    if let Some(t) = reinterpret_type {
        arr = reinterpret(arr, t).map_err(to_err)?;
    }
    Ok((to_scalars(arr), false))
}

// an integer array of the given type, from json integers. floats are
// accepted when they have no fractional part, e.g. -9999.0.
fn int_array(values: &[&Value], data_type: &DataType) -> Result<ArrayRef, String> {
    let invalid = |v: &Value| format!("invalid fill value {v} for type {data_type}");
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };

    let arr: ArrayRef = if data_type.is_unsigned_integer() {
        let ints = values
            .iter()
            .map(|v| {
                v.as_u64()
                    .or_else(|| {
                        v.as_f64()
                            .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                            .map(|f| f as u64)
                    })
                    .ok_or_else(|| invalid(v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Arc::new(UInt64Array::from(ints))
    } else {
        let ints = values
            .iter()
            .map(|v| {
                v.as_i64()
                    .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                    .ok_or_else(|| invalid(v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Arc::new(Int64Array::from(ints))
    };
    cast_with_options(&arr, data_type, &options)
        .map_err(|_| format!("fill values {values:?} don't fit in type {data_type}"))
}

// a float from a json number, a special value string ("NaN", "Infinity",
// "-Infinity"), or the base64 of the 8 little endian bytes of an f64, which
// is how xarray writes float fill values in zarr v3.
fn json_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            s => {
                let bytes = BASE64_STANDARD.decode(s).ok()?;
                Some(f64::from_le_bytes(bytes.try_into().ok()?))
            }
        },
        _ => None,
    }
}

// an optional float attribute, a single element list is accepted too since
// netcdf attributes are often stored that way.
fn attr_f64(attrs: &Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    let value = match attrs.get(key) {
        None => return Ok(None),
        Some(Value::Array(values)) if values.len() == 1 => &values[0],
        Some(v) => v,
    };
    value
        .as_f64()
        .map(Some)
        .ok_or_else(|| format!("invalid {key} {value}"))
}

//***************************************************
// Transforms the data read from an array, based on the CF and
// NetCDF conventions in the array's attributes. the transformations
// are applied in the same order as xarray.
//***************************************************
pub(crate) struct DataTransformer {
    // the type of the arrays read from the store, before any transformation.
    raw_type: DataType,
    nullable: bool,

    // _Unsigned, the signed or unsigned integer type to reinterpret as.
    reinterpret_type: Option<DataType>,

    // _FillValue, missing_value and the valid range.
    mask: Option<DataMask>,

    // dtype "bool", integers become booleans.
    to_bool: bool,

    scale_factor: Option<f64>,
    add_offset: Option<f64>,

    // any time or duration convertion from attributes.
    time: Option<TimeConversion>,
}

impl DataTransformer {
    // the field has the raw type of the array, as read from the store.
    pub(crate) fn new(field: &Field, attrs: &Map<String, Value>) -> ZarrQueryResult<Self> {
        let raw_type = field.data_type().clone();
        let name = field.name();
        let invalid = |msg: &str| ZarrQueryError::InvalidMetadata(format!("Array {name}: {msg}"));

        let reinterpret_type = match attrs.get("_Unsigned") {
            None => None,
            Some(v) => reinterpret_type(&raw_type, v).map_err(|msg| invalid(&msg))?,
        };

        let mask = DataMask::new(attrs, &raw_type, reinterpret_type.as_ref())
            .map_err(|msg| invalid(&msg))?;

        // bool conversion and scale/offsets, which are mutually exclusive
        // so checked together.
        let to_bool = attrs.get("dtype").and_then(Value::as_str) == Some("bool");
        let scale_factor = attr_f64(attrs, "scale_factor").map_err(|msg| invalid(&msg))?;
        let add_offset = attr_f64(attrs, "add_offset").map_err(|msg| invalid(&msg))?;

        // attributes that transform an int or float to a time or duration.
        let units = attrs.get("units").and_then(Value::as_str);
        let is_timedelta = attrs
            .get("dtype")
            .and_then(Value::as_str)
            .is_some_and(|d| d.starts_with("timedelta64"));
        let time = match units {
            Some(units) if units.to_lowercase().contains(" since ") => {
                let calendar = attrs.get("calendar").and_then(Value::as_str);
                Some(TimeConversion::time_since(units, calendar)?)
            }
            Some(units) if is_timedelta => Some(TimeConversion::duration(units)?),
            _ => None,
        };

        let transformer = Self {
            raw_type,
            nullable: field.is_nullable(),
            reinterpret_type,
            mask,
            to_bool,
            scale_factor,
            add_offset,
            time,
        };
        transformer.validate().map_err(|msg| invalid(&msg))?;
        Ok(transformer)
    }

    // final validation on the build transformer.
    fn validate(&self) -> Result<(), String> {
        let unpack = self.scale_factor.is_some() || self.add_offset.is_some();
        let any = self.reinterpret_type.is_some()
            || self.mask.is_some()
            || self.to_bool
            || unpack
            || self.time.is_some();
        let numeric = self.raw_type.is_integer() || self.raw_type.is_floating();
        if any && !numeric {
            return Err(format!(
                "CF attributes are only supported on int and float arrays, not {}",
                self.raw_type
            ));
        }

        if self.to_bool {
            if !self.packed_type().is_integer() {
                return Err("dtype bool is only supported on int arrays".into());
            }
            if unpack || self.time.is_some() {
                return Err("dtype bool can't be combined with unpacking or time units".into());
            }
        }

        // the values the time conversion sees, floats can be NaN.
        let time_input_is_float = unpack || self.packed_type().is_floating();
        let can_be_null = self.mask.is_some() || (self.time.is_some() && time_input_is_float);
        if can_be_null && !self.nullable {
            return Err("the attributes can produce nulls, but the field isn't nullable".into());
        }
        Ok(())
    }

    fn packed_type(&self) -> &DataType {
        self.reinterpret_type.as_ref().unwrap_or(&self.raw_type)
    }

    fn output_type(&self) -> DataType {
        if let Some(time) = &self.time {
            time.data_type()
        } else if self.to_bool {
            DataType::Boolean
        } else if self.scale_factor.is_some() || self.add_offset.is_some() {
            DataType::Float64
        } else {
            self.packed_type().clone()
        }
    }

    // the field as it is after the transformations, from the raw field.
    pub(crate) fn output_field(&self, field: &Field) -> Field {
        field.clone().with_data_type(self.output_type())
    }

    // the array must be of the raw type, i.e. read from the array this
    // transformer was built for.
    pub(crate) fn apply(&self, mut arr: ArrayRef) -> ZarrQueryResult<ArrayRef> {
        if arr.data_type() != &self.raw_type {
            return Err(ZarrQueryError::InvalidCompute(format!(
                "Expected an array of type {} to transform, got {}",
                self.raw_type,
                arr.data_type()
            )));
        }

        if let Some(t) = &self.reinterpret_type {
            arr = reinterpret(arr, t)?;
        }
        if let Some(mask) = &self.mask {
            arr = mask.apply(arr)?;
        }
        if self.to_bool {
            arr = cast(&arr, &DataType::Boolean)?;
        }
        if self.scale_factor.is_some() || self.add_offset.is_some() {
            let scale = self.scale_factor.unwrap_or(1.0);
            let offset = self.add_offset.unwrap_or(0.0);
            let floats = cast(&arr, &DataType::Float64)?;
            let unpacked: Float64Array = floats
                .as_primitive::<Float64Type>()
                .unary(|v| v * scale + offset);
            arr = Arc::new(unpacked);
        }
        if let Some(time) = &self.time {
            arr = time.convert(arr)?;
        }
        Ok(arr)
    }
}

//***************************************************
// The interface to an array in a zarr store.
//***************************************************
struct ArrayInterface<T: AsyncReadableListableStorageTraits + ?Sized> {
    name: String,
    arr: Arc<Array<T>>,
    transformer: Arc<DataTransformer>,
    chk_index: Vec<u64>,

    // the real chunk shape, adjusted for the array edges, and a flag
    // to specify if the chunk idx maps to an edge chunk.
    chunk_shape: Vec<u64>,
    is_edge: bool,
}

enum BytesFromArray {
    // the static lifetime is there because the ArrayBytes comes from calling
    // into_owned() on the ArrayBytes we get from reading the zarr data.
    Decoded(ArrayBytes<'static>),
    Encoded(Option<Bytes>),
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ArrayInterface<T> {
    async fn read_bytes(&self) -> ZarrQueryResult<BytesFromArray> {
        // handling edges is easier if we use zarrs built in method, but it
        // reads and decodes, so we don't split between runtimes here.
        if self.is_edge {
            let ranges: Vec<_> = self.chunk_shape.iter().map(|c| 0..*c).collect();
            let array_subset = ArraySubset::new_with_ranges(&ranges);
            let data = self
                .arr
                .async_retrieve_chunk_subset::<ArrayBytes>(&self.chk_index, &array_subset)
                .await?;
            Ok(BytesFromArray::Decoded(data.into_owned()))
        } else {
            let data = self
                .arr
                .async_retrieve_encoded_chunk(&self.chk_index)
                .await?;
            Ok(BytesFromArray::Encoded(data))
        }
    }

    // separate from the read so that decoding can happen on the compute
    // runtime while other reads are still in flight.
    fn decode_data(&self, bytes: BytesFromArray) -> ZarrQueryResult<ArrayRef> {
        let decoded_bytes = match bytes {
            BytesFromArray::Encoded(Some(bytes)) => {
                let chunk_shape = self.arr.chunk_shape(&self.chk_index)?;
                self.arr.codecs().decode(
                    // move the raw bytes into the Vec (reusing the allocation)
                    // rather than copying them; the chain decode wants raw bytes
                    // (ArrayBytesRaw = Cow<[u8]>), which a Vec coerces into.
                    Vec::from(bytes).into(),
                    &chunk_shape,
                    self.arr.data_type(),
                    self.arr.fill_value(),
                    &CodecOptions::default(),
                )?
            }
            BytesFromArray::Encoded(None) => ArrayBytes::new_fill_value(
                self.arr.data_type(),
                self.chunk_shape.num_elements(),
                self.arr.fill_value(),
            )?,
            BytesFromArray::Decoded(bytes) => bytes,
        };
        let raw = array_bytes_to_arrow(self.arr.data_type(), decoded_bytes)?;
        self.transformer.apply(raw)
    }
}

type ZarrReceiver<T> = Receiver<(ZarrQueryResult<BytesFromArray>, ArrayInterface<T>)>;

// The mutable state for reading chunks, owned by the reader so that the
// store interface itself can be shared.
pub(crate) struct ChunkReadState<T: AsyncReadableListableStorageTraits + ?Sized> {
    join_set: JoinSet<()>,
    metrics: ZarrMetrics,

    // the prefetched reads, with their chunk index and columns.
    state: Option<(ZarrReceiver<T>, Vec<u64>, Vec<String>)>,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized> ChunkReadState<T> {
    pub(crate) fn new(metrics: ZarrMetrics) -> Self {
        Self {
            join_set: JoinSet::new(),
            state: None,
            metrics,
        }
    }

    pub(crate) fn metrics(&self) -> &ZarrMetrics {
        &self.metrics
    }
}

// A future resolving to a store interface, shared between the streams of a
// scan. the error is wrapped in an Arc because a shared future's output must
// be cloneable.
pub(crate) type SharedZarrStoreInterface<T> =
    Shared<BoxFuture<'static, Result<Arc<ZarrStoreInterface<T>>, Arc<ZarrQueryError>>>>;

//***************************************************
// The interface to the zarr store
//***************************************************
pub(crate) struct ZarrStoreInterface<T: AsyncReadableListableStorageTraits + ?Sized> {
    schema_ref: SchemaRef,
    coords: HashMap<String, BroadcastableCoord>,
    grid: ChunkGrid,

    // non 1D coordinate arrays only, with the transformations of their
    // attributes. the coordinates are transformed once, when they're read.
    arrays: HashMap<String, (Arc<Array<T>>, Arc<DataTransformer>)>,

    // offloads reading data to a dedicated i/o runtime.
    io_runtime: IoRuntime,
}

impl<T: AsyncReadableListableStorageTraits + ?Sized + 'static> ZarrStoreInterface<T> {
    pub(crate) async fn new(
        store: Arc<T>,
        prefix: Option<String>,
        schema_ref: SchemaRef,
        metrics: &ZarrMetrics,
    ) -> ZarrQueryResult<Self> {
        if schema_ref.fields().is_empty() {
            return Err(ZarrQueryError::InvalidColumnRequest(
                "No arrays to open in the zarr store".into(),
            ));
        }

        // some remote stores (e.g. s3) don't work off of the url, just the
        // bucket, so the path to the zarr store is provided separately.
        let prefix = match prefix {
            Some(prefix) => format!("/{prefix}"),
            None => "/".to_string(),
        };
        let cols = schema_ref.fields().column_names();

        // each coordinate's name, field, array, transformer, and its own 1D
        // chunk grid.
        let mut coord_arrays: Vec<(String, FieldRef, Array<T>, DataTransformer, ChunkGrid)> =
            Vec::new();

        let mut data_arrays: HashMap<String, (Arc<Array<T>>, Arc<DataTransformer>)> =
            HashMap::new();
        let mut data_meta: Option<DataArraysMeta> = None;

        // for each col in the schema, determine if it's a 1d coord or not.
        // data arrays are stored for future reads, but coords are fully read
        // and cached, just outside of the below for loop.
        for col in &cols {
            let path = PathBuf::from(&prefix)
                .join(col)
                .to_str()
                .ok_or(ZarrQueryError::InvalidCompute(
                    "Could not form path from group and column name".into(),
                ))?
                .to_string();
            let arr = Array::async_open(store.clone(), &path).await?;
            let (_, field) = schema_ref.fields().find(col).ok_or_else(|| {
                ZarrQueryError::InvalidColumnRequest(format!("Array {col} is not in the schema"))
            })?;
            let transformer = data_transformer(field, &arr)?;
            if is_1d_coordinate(col, &arr)? {
                let grid = ChunkGrid::from_array(&arr)?;
                coord_arrays.push((col.to_string(), field.clone(), arr, transformer, grid));
            } else {
                match &data_meta {
                    Some(meta) => meta.validate(col, &arr)?,
                    None => data_meta = Some(DataArraysMeta::new(&arr)?),
                }
                data_arrays.insert(col.to_string(), (Arc::new(arr), Arc::new(transformer)));
            }
        }

        let reads =
            coord_arrays
                .into_iter()
                .map(|(name, field, arr, transformer, grid)| async move {
                    // here the full 1d coordinate data is read, and transformed.
                    let values = transformer.apply(read_full_array(&arr).await?)?;
                    Ok::<_, ZarrQueryError>((name, field, values, grid))
                });
        let read_coords = metrics
            .time_io(futures::future::try_join_all(reads))
            .await?;

        let (coords, grid) = match data_meta {
            Some(meta) => {
                let coords = read_coords
                    .into_iter()
                    .map(|(name, field, values, _)| {
                        let position = meta.validate_coord(&name, &values)?;
                        Ok((
                            name,
                            BroadcastableCoord {
                                position,
                                field,
                                values,
                            },
                        ))
                    })
                    .collect::<ZarrQueryResult<_>>()?;
                (coords, meta.grid)
            }
            None => coords_only_meta(read_coords),
        };

        // this runtime will handle the i/o. i/o tasks spawned in that runtime
        // will not share a thead pool with other (probably compute heavy,
        // blocking) tasks.
        let io_runtime = IoRuntime::try_new()?;

        Ok(Self {
            schema_ref,
            arrays: data_arrays,
            coords,
            grid,
            io_runtime,
        })
    }

    pub(crate) fn schema_ref(&self) -> &SchemaRef {
        &self.schema_ref
    }

    pub(crate) fn chunk_grid_shape(&self) -> &[u64] {
        &self.grid.grid_shape
    }

    fn is_coord(&self, name: &str) -> bool {
        self.coords.contains_key(name)
    }

    fn is_data_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    // whether every column is a 1D coordinate.
    fn only_coords<S: AsRef<str>>(&self, cols: impl IntoIterator<Item = S>) -> bool {
        cols.into_iter().all(|col| self.is_coord(col.as_ref()))
    }

    // the columns that are data arrays, in order.
    fn data_cols<S: AsRef<str>>(&self, cols: impl IntoIterator<Item = S>) -> Vec<String> {
        cols.into_iter()
            .filter(|col| self.is_data_array(col.as_ref()))
            .map(|col| col.as_ref().to_string())
            .collect()
    }

    fn get_array_interfaces(
        &self,
        cols: &[String],
        chk_idx: &[u64],
    ) -> ZarrQueryResult<Vec<ArrayInterface<T>>> {
        let chunk_shape = self.grid.chunk_shape(chk_idx);
        let is_edge = self.grid.is_edge(chk_idx);
        cols.iter()
            .map(|col| {
                let (arr, transformer) = self
                    .arrays
                    .get(col)
                    .ok_or_else(|| ZarrQueryError::array_not_in_store(col))?
                    .clone();
                Ok(ArrayInterface {
                    name: col.to_string(),
                    arr,
                    transformer,
                    chk_index: chk_idx.to_vec(),
                    chunk_shape: chunk_shape.clone(),
                    is_edge,
                })
            })
            .collect()
    }

    // spawns a read task per array interface on the i/o runtime, returning the
    // receiver that yields each (read result, interface) pair as it completes.
    fn spawn_reads(
        &self,
        interfaces: Vec<ArrayInterface<T>>,
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrReceiver<T> {
        let (tx, rx) = tokio::sync::mpsc::channel(interfaces.len());
        for arr_interface in interfaces {
            let tx = tx.clone();
            let metrics = read_state.metrics.clone();
            let io_task = async move {
                let b = metrics.time_io(arr_interface.read_bytes()).await;
                let _ = tx.send((b, arr_interface)).await;
            };
            read_state
                .join_set
                .spawn_on(io_task, self.io_runtime.handle());
        }
        rx
    }

    // decodes each array as its read completes, which overlaps decoding one
    // array with the still in flight reads of the others.
    async fn drain_decode(
        mut rx: ZarrReceiver<T>,
        metrics: &ZarrMetrics,
    ) -> ZarrQueryResult<HashMap<String, ArrayRef>> {
        let mut data = HashMap::new();
        while let Some((bytes, arr_interface)) = rx.recv().await {
            let bytes = bytes?;
            let arr = metrics.time_decode(|| arr_interface.decode_data(bytes))?;
            data.insert(arr_interface.name, arr);
        }
        Ok(data)
    }

    // reads a chunk without using or setting any prefetched reads.
    async fn get_chunk_uncached(
        &self,
        cols: &[String],
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        self.get_chunk(cols, chk_idx.to_vec(), false, None, read_state)
            .await
    }

    // returns the requested columns for a chunk. the 1D coordinates are always
    // broadcast over every dimension, so that the number of rows is the same
    // regardless of the columns requested. this can also trigger the next
    // cbunk to be prefetched, optionally.
    pub(crate) async fn get_chunk(
        &self,
        cols: &[String],
        chk_idx: Vec<u64>,
        use_cached_value: bool,
        next_chunk_idx: Option<Vec<u64>>,
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<ZarrInMemoryChunk> {
        if cols.is_empty() {
            return Err(ZarrQueryError::InvalidColumnRequest(
                "No columns when polling zarr store for chunks".into(),
            ));
        }

        let mut data_cols = Vec::new();
        let mut coords = Vec::new();
        for col in cols {
            if self.is_data_array(col) {
                data_cols.push(col.clone());
            } else if let Some(coord) = self.coords.get(col) {
                coords.push(coord);
            } else {
                return Err(ZarrQueryError::array_not_in_store(col));
            }
        }
        let coords_batch = coords.as_slice().broadcast(&self.grid.ranges(&chk_idx))?;

        let mut chunk = ZarrInMemoryChunk::new();
        chunk.add_batch(&coords_batch);

        if !data_cols.is_empty() {
            // either use the reads prefetched by the previous call, or spawn
            // them now.
            let rx = if use_cached_value && read_state.state.is_some() {
                let (rx, cached_idx, cached_cols) = read_state
                    .state
                    .take()
                    .expect("cache hit implies state is present");
                if cached_idx != chk_idx {
                    return Err(ZarrQueryError::InvalidCompute(
                        "Cached zarr chunk index doesn't match requested chunk index".into(),
                    ));
                }
                if cached_cols != data_cols {
                    return Err(ZarrQueryError::InvalidCompute(
                        "Cached zarr chunk columns don't match requested columns".into(),
                    ));
                }
                rx
            } else {
                let interfaces = self.get_array_interfaces(&data_cols, &chk_idx)?;
                self.spawn_reads(interfaces, read_state)
            };
            for (name, arr) in Self::drain_decode(rx, &read_state.metrics).await? {
                chunk.add_data(name, arr);
            }

            // prefetch the next chunk, so that its data is fetched while other
            // operations run between now and the next call.
            if let Some(next_chunk_idx) = next_chunk_idx {
                let interfaces = self.get_array_interfaces(&data_cols, &next_chunk_idx)?;
                let rx = self.spawn_reads(interfaces, read_state);
                read_state.state = Some((rx, next_chunk_idx, data_cols));
            }
        }

        Ok(chunk)
    }

    // evaluates each of the filter's predicates on a record batch with only the
    // predicate's columns, stopping at the first one that no row passes. the 1D
    // coordinates are only broadcast over the dimensions a predicate needs. returns
    // None if the chunk can be skipped, otherwise the full chunk columns that
    // were read.
    pub(crate) async fn evaluate_filter(
        &self,
        filter: &ZarrChunkFilter,
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<Option<ZarrInMemoryChunk>> {
        // the predicates on only 1D coordinates are cheap, so they go first,
        // then the predicates with fewer columns.
        let mut keys: Vec<&BTreeSet<String>> = filter.keys().collect();
        keys.sort_by_key(|key| (!self.only_coords(key.iter()), key.len()));

        let chunk_ranges = self.grid.ranges(chk_idx);
        let mut read_chunk = ZarrInMemoryChunk::new();
        for key in keys {
            let batch = if self.only_coords(key.iter()) {
                let coords: Vec<&BroadcastableCoord> =
                    key.iter().map(|col| &self.coords[col]).collect();
                coords.as_slice().broadcast_chunk(&chunk_ranges)?
            } else {
                let cols: Vec<String> = key.iter().cloned().collect();
                let chunk = self.get_chunk_uncached(&cols, chk_idx, read_state).await?;
                let batch = chunk.to_record_batch()?;

                // the columns read are kept so that they don't need to be read
                // again after the filter.
                read_chunk.combine(chunk);
                batch
            };

            if !filter.evaluate(&batch)? {
                return Ok(None);
            }
        }

        Ok(Some(read_chunk))
    }

    // evaluates the chunk-pruning expression on the min/max of each pruning
    // column. the min/max of a 1D coordinate is the same whether it's
    // broadcast or not, so they are computed from the raw coordinate values.
    pub(crate) async fn keep_chunk(
        &self,
        pruning: &ChunkPruning,
        chk_idx: &[u64],
        read_state: &mut ChunkReadState<T>,
    ) -> ZarrQueryResult<bool> {
        let data_cols = self.data_cols(pruning.schema.fields().iter_column_names());
        let mut chunk = if data_cols.is_empty() {
            ZarrInMemoryChunk::new()
        } else {
            self.get_chunk_uncached(&data_cols, chk_idx, read_state)
                .await?
        };

        let ranges = self.grid.ranges(chk_idx);
        for field in pruning.schema.fields() {
            let name = field.name();
            if let Some(coord) = self.coords.get(name) {
                chunk.add_data(name.clone(), coord.slice(&ranges)?);
            }
        }

        let min_max = chunk.to_min_max_record_batch(&pruning.schema)?;
        chunk_kept(&pruning.expr, &min_max)
    }
}

//***************************************************
// A structure to accumulate zarr array data until we can output
// the whole chunk as a record batch.
//***************************************************
pub(crate) struct ZarrInMemoryChunk {
    data: HashMap<String, ArrayRef>,
}

impl ZarrInMemoryChunk {
    pub(crate) fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn add_data(&mut self, arr_name: String, data: ArrayRef) {
        self.data.insert(arr_name, data);
    }

    fn add_batch(&mut self, batch: &RecordBatch) {
        for (field, arr) in batch.schema().fields().iter().zip(batch.columns()) {
            self.add_data(field.name().clone(), arr.clone());
        }
    }

    pub(crate) fn combine(&mut self, other: ZarrInMemoryChunk) {
        self.data.extend(other.data);
    }

    pub(crate) fn get(&self, arr_name: &str) -> Option<&ArrayRef> {
        self.data.get(arr_name)
    }

    // gathers the chunk's arrays in schema-field order, erroring if any
    // field's array is missing from the chunk.
    fn columns_in_schema_order(&self, schema: &Schema) -> ZarrQueryResult<Vec<ArrayRef>> {
        schema
            .fields()
            .iter()
            .map(|f| {
                self.data
                    .get(f.name())
                    .cloned()
                    .ok_or_else(|| ZarrQueryError::array_not_in_chunk(f.name()))
            })
            .collect()
    }

    // a record batch with all the arrays in the chunk, in no particular order.
    fn to_record_batch(&self) -> ZarrQueryResult<RecordBatch> {
        Ok(RecordBatch::try_from_iter(
            self.data.iter().map(|(k, v)| (k.clone(), v.clone())),
        )?)
    }

    pub(crate) fn into_record_batch(self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let columns = self.columns_in_schema_order(schema)?;
        Ok(RecordBatch::try_new(schema.clone(), columns)?)
    }

    // a 2-row record batch holding, per column, that column's min (row 0) and
    // max (row 1) in schema-field order.
    fn to_min_max_record_batch(&self, schema: &SchemaRef) -> ZarrQueryResult<RecordBatch> {
        let min_max = self
            .columns_in_schema_order(schema)?
            .iter()
            .zip(schema.fields())
            .map(|(arr, f)| Ok((f.name().clone(), min_max_array(arr)?)))
            .collect::<ZarrQueryResult<Vec<_>>>()?;
        Ok(RecordBatch::try_from_iter(min_max)?)
    }
}

//***************************************************
// A resolved chunk-pruning expression and the subset schema it reads.
// This is currently used for pruning based on spatial joins, and the logic
// relies on the pruning only depending on the min/max of all the columns.
//***************************************************
pub(crate) struct ChunkPruning {
    expr: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl ChunkPruning {
    // Selects the columns `expr` references from `projected_schema` (by name)
    // into a subset schema, and reassigns the expression's column indices to it
    // so it evaluates against a min/max batch built from that subset.
    pub(crate) fn try_new(
        expr: Arc<dyn PhysicalExpr>,
        projected_schema: &SchemaRef,
    ) -> ZarrQueryResult<Self> {
        let mut indices = collect_columns(&expr)
            .iter()
            .map(|c| projected_schema.index_of(c.name()))
            .collect::<Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();

        let schema = Arc::new(projected_schema.project(&indices)?);
        let expr = reassign_expr_columns(expr, &schema)?;
        Ok(Self { expr, schema })
    }
}

// Evaluates a chunk-pruning expression against a chunk's min/max record batch,
// returning whether the chunk should be kept (a null verdict keeps it).
fn chunk_kept(expr: &Arc<dyn PhysicalExpr>, min_max: &RecordBatch) -> ZarrQueryResult<bool> {
    let value = expr.evaluate(min_max)?;
    match value {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(keep))) => Ok(keep),
        ColumnarValue::Scalar(ScalarValue::Boolean(None)) => Ok(true),
        other => Err(ZarrQueryError::InvalidCompute(format!(
            "chunk-pruning expression must return a boolean scalar, got {other:?}"
        ))),
    }
}

// Returns a length-2 array `[min, max]` of the same numeric type as `array`.
// `None` min/max (empty or all-null input) become nulls.
fn min_max_array(array: &ArrayRef) -> ZarrQueryResult<ArrayRef> {
    macro_rules! min_max {
        ($ty:ty) => {{
            let a = array
                .as_any()
                .downcast_ref::<PrimitiveArray<$ty>>()
                .expect("array data type matched");
            Arc::new(PrimitiveArray::<$ty>::from(vec![
                arrow_min(a),
                arrow_max(a),
            ])) as ArrayRef
        }};
    }

    let out = match array.data_type() {
        DataType::Float64 => min_max!(Float64Type),
        DataType::Float32 => min_max!(Float32Type),
        DataType::Int64 => min_max!(Int64Type),
        DataType::Int32 => min_max!(Int32Type),
        DataType::Int16 => min_max!(Int16Type),
        DataType::Int8 => min_max!(Int8Type),
        DataType::UInt64 => min_max!(UInt64Type),
        DataType::UInt32 => min_max!(UInt32Type),
        DataType::UInt16 => min_max!(UInt16Type),
        DataType::UInt8 => min_max!(UInt8Type),
        other => {
            return Err(ZarrQueryError::InvalidColumnRequest(format!(
                "min/max not supported for data type {other}"
            )))
        }
    };
    Ok(out)
}

fn dim_names<T: ?Sized>(arr: &Array<T>) -> ZarrQueryResult<Option<Vec<String>>> {
    arr.dimension_names()
        .clone()
        .map(|names| {
            names
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or(ZarrQueryError::InvalidMetadata(
                    "Null dimension names not supported".into(),
                ))
        })
        .transpose()
}

// the transformer for an array, from the array's raw type and attributes.
// the field is from the schema, so it has the output type, which has to
// match what the transformer produces.
fn data_transformer<T: ?Sized>(field: &Field, arr: &Array<T>) -> ZarrQueryResult<DataTransformer> {
    let raw_field = Field::new(
        field.name(),
        arrow_type(arr.data_type())?,
        field.is_nullable(),
    );
    let transformer = DataTransformer::new(&raw_field, arr.attributes())?;
    let output_type = transformer.output_field(&raw_field).data_type().clone();
    if &output_type != field.data_type() {
        return Err(ZarrQueryError::InvalidMetadata(format!(
            "Array {} has type {output_type} after transformations, but the schema expects {}",
            field.name(),
            field.data_type()
        )));
    }
    Ok(transformer)
}

fn is_1d_coordinate<T: ?Sized>(name: &str, arr: &Array<T>) -> ZarrQueryResult<bool> {
    Ok(is_1d_coordinate_from_meta(
        name,
        dim_names(arr)?.as_deref(),
        arr.dimensionality(),
    ))
}

// a 1-D coordinate is a 1-D array that names its own single dimension.
// everything else, including already broadcast N-D coordinates, is a data array.
pub(crate) fn is_1d_coordinate_from_meta(
    name: &str,
    dim_names: Option<&[String]>,
    dimensionality: usize,
) -> bool {
    dimensionality == 1 && matches!(dim_names, Some([dim]) if dim == name)
}

async fn read_full_array<T: AsyncReadableListableStorageTraits + ?Sized + 'static>(
    arr: &Array<T>,
) -> ZarrQueryResult<ArrayRef> {
    let bytes = arr
        .async_retrieve_array_subset::<ArrayBytes>(&arr.subset_all())
        .await?;
    array_bytes_to_arrow(arr.data_type(), bytes)
}

fn array_bytes_to_arrow(t: &zDataType, bytes: ArrayBytes<'_>) -> ZarrQueryResult<ArrayRef> {
    macro_rules! to_array_ref {
        ($array_t: ty, $prim_type: ty) => {{
            let arr: $array_t = <$prim_type>::from_array_bytes(t, bytes)?.into();
            Ok(Arc::new(arr) as ArrayRef)
        }};
    }

    if t.is::<BoolDataType>() {
        to_array_ref!(BooleanArray, bool)
    } else if t.is::<UInt8DataType>() {
        to_array_ref!(PrimitiveArray<UInt8Type>, u8)
    } else if t.is::<UInt16DataType>() {
        to_array_ref!(PrimitiveArray<UInt16Type>, u16)
    } else if t.is::<UInt32DataType>() {
        to_array_ref!(PrimitiveArray<UInt32Type>, u32)
    } else if t.is::<UInt64DataType>() {
        to_array_ref!(PrimitiveArray<UInt64Type>, u64)
    } else if t.is::<Int8DataType>() {
        to_array_ref!(PrimitiveArray<Int8Type>, i8)
    } else if t.is::<Int16DataType>() {
        to_array_ref!(PrimitiveArray<Int16Type>, i16)
    } else if t.is::<Int32DataType>() {
        to_array_ref!(PrimitiveArray<Int32Type>, i32)
    } else if t.is::<Int64DataType>() {
        to_array_ref!(PrimitiveArray<Int64Type>, i64)
    } else if t.is::<Float32DataType>() {
        to_array_ref!(PrimitiveArray<Float32Type>, f32)
    } else if t.is::<Float64DataType>() {
        to_array_ref!(PrimitiveArray<Float64Type>, f64)
    } else if t.is::<StringDataType>() {
        to_array_ref!(StringArray, String)
    } else if let Some(dt) = t.downcast_ref::<NumpyDateTime64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        match dt.unit {
            NumpyTimeUnit::Second => to_array_ref!(TimestampSecondArray, i64),
            NumpyTimeUnit::Millisecond => to_array_ref!(TimestampMillisecondArray, i64),
            NumpyTimeUnit::Microsecond => to_array_ref!(TimestampMicrosecondArray, i64),
            NumpyTimeUnit::Nanosecond => to_array_ref!(TimestampNanosecondArray, i64),
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported datetime64 unit {} from zarr metadata",
                dt.unit
            ))),
        }
    } else if let Some(dt) = t.downcast_ref::<NumpyTimeDelta64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        match dt.unit {
            NumpyTimeUnit::Second => to_array_ref!(DurationSecondArray, i64),
            NumpyTimeUnit::Millisecond => to_array_ref!(DurationMillisecondArray, i64),
            NumpyTimeUnit::Microsecond => to_array_ref!(DurationMicrosecondArray, i64),
            NumpyTimeUnit::Nanosecond => to_array_ref!(DurationNanosecondArray, i64),
            _ => Err(ZarrQueryError::InvalidType(format!(
                "Unsupported timedelta64 unit {} from zarr metadata",
                dt.unit
            ))),
        }
    } else {
        Err(ZarrQueryError::InvalidType(format!(
            "Unsupported type {t} from zarr metadata"
        )))
    }
}

// the arrow type of the arrays read from a zarr array of the given type, i.e.
// the type that `array_bytes_to_arrow` produces.
pub(crate) fn arrow_type(t: &zDataType) -> ZarrQueryResult<DataType> {
    if t.is::<BoolDataType>() {
        Ok(DataType::Boolean)
    } else if t.is::<UInt8DataType>() {
        Ok(DataType::UInt8)
    } else if t.is::<UInt16DataType>() {
        Ok(DataType::UInt16)
    } else if t.is::<UInt32DataType>() {
        Ok(DataType::UInt32)
    } else if t.is::<UInt64DataType>() {
        Ok(DataType::UInt64)
    } else if t.is::<Int8DataType>() {
        Ok(DataType::Int8)
    } else if t.is::<Int16DataType>() {
        Ok(DataType::Int16)
    } else if t.is::<Int32DataType>() {
        Ok(DataType::Int32)
    } else if t.is::<Int64DataType>() {
        Ok(DataType::Int64)
    } else if t.is::<Float32DataType>() {
        Ok(DataType::Float32)
    } else if t.is::<Float64DataType>() {
        Ok(DataType::Float64)
    } else if t.is::<StringDataType>() {
        Ok(DataType::Utf8)
    } else if let Some(dt) = t.downcast_ref::<NumpyDateTime64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        Ok(DataType::Timestamp(arrow_time_unit(t, dt.unit)?, None))
    } else if let Some(dt) = t.downcast_ref::<NumpyTimeDelta64DataType>() {
        ensure_unit_scale(t, dt.scale_factor.get())?;
        Ok(DataType::Duration(arrow_time_unit(t, dt.unit)?))
    } else {
        Err(ZarrQueryError::InvalidType(format!(
            "Unsupported type {t} from zarr metadata"
        )))
    }
}

fn arrow_time_unit(t: &zDataType, unit: NumpyTimeUnit) -> ZarrQueryResult<TimeUnit> {
    match unit {
        NumpyTimeUnit::Second => Ok(TimeUnit::Second),
        NumpyTimeUnit::Millisecond => Ok(TimeUnit::Millisecond),
        NumpyTimeUnit::Microsecond => Ok(TimeUnit::Microsecond),
        NumpyTimeUnit::Nanosecond => Ok(TimeUnit::Nanosecond),
        _ => Err(ZarrQueryError::InvalidType(format!(
            "Unsupported unit {unit} for type {t} from zarr metadata"
        ))),
    }
}

// arrow's timestamp/duration types can't represent a scale factor other than 1,
// so we reject anything else rather than silently reading wrong values.
fn ensure_unit_scale(t: &zDataType, scale_factor: u32) -> ZarrQueryResult<()> {
    if scale_factor != 1 {
        return Err(ZarrQueryError::InvalidType(format!(
            "Unsupported scale factor {scale_factor} for type {t} from zarr metadata"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod zarr_store_interface_tests {
    use super::*;
    use crate::test_utils::get_local_zarr_store;

    #[tokio::test]
    async fn data_and_coords_get_chunk_test() {
        let (wrapper, schema) =
            get_local_zarr_store(true, 0.0, "store_interface_data_and_coords").await;
        let store = wrapper.get_store();

        let names: Vec<String> = vec!["data".into(), "lat".into(), "lon".into()];
        let metrics = ZarrMetrics::disconnected();
        let interface = ZarrStoreInterface::new(store, None, schema, &metrics)
            .await
            .unwrap();
        let mut read_state = ChunkReadState::new(metrics.clone());

        let batch = interface
            .get_chunk(
                &["lat".to_string()],
                vec![0, 0],
                false,
                None,
                &mut read_state,
            )
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 9);

        let batch = interface
            .get_chunk(&names, vec![0, 0], false, None, &mut read_state)
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.num_rows(), 9);
    }

    #[tokio::test]
    async fn coord_only_get_chunk_test() {
        let (wrapper, schema) = get_local_zarr_store(true, 0.0, "store_interface_coord_only").await;
        let store = wrapper.get_store();

        let names: Vec<String> = vec!["lat".into()];
        let schema = Arc::new(schema.project(&[schema.index_of("lat").unwrap()]).unwrap());
        let metrics = ZarrMetrics::disconnected();
        let interface = ZarrStoreInterface::new(store, None, schema, &metrics)
            .await
            .unwrap();
        let mut read_state = ChunkReadState::new(metrics.clone());

        let batch = interface
            .get_chunk(&names, vec![0], false, None, &mut read_state)
            .await
            .unwrap()
            .to_record_batch()
            .unwrap();
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 3);
    }
}
