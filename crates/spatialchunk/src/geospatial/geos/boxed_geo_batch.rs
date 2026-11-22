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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{ArrayRef, RecordBatch};
use datafusion::error::Result;
use datafusion::physical_plan::{PhysicalExpr, SendableRecordBatchStream};
use futures::Stream;
use geo_types::Rect;

use super::raw_geo::RawGeo;

// Packages a record batch and all the data necessary to interact with the
// geometry objects in one of the columns.
pub(crate) struct BBoxedGeoBatch {
    pub(crate) batch: RecordBatch,
    pub(crate) geo_array: ArrayRef,
    pub(crate) rects: Vec<Rect<f32>>,
    pub(crate) geo_ids: Vec<usize>,
}

impl BBoxedGeoBatch {
    // Splits up to n (rect, geo_id) pairs off the back of the batch and returns
    // them. rects and geo_ids stay parallel within the chunk.
    pub(crate) fn pop_chunk(&mut self, n: usize) -> (Vec<Rect<f32>>, Vec<usize>) {
        let start = self.geo_ids.len().saturating_sub(n);
        let rects = self.rects.split_off(start);
        let geo_ids = self.geo_ids.split_off(start);
        (rects, geo_ids)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.geo_ids.is_empty()
    }
}

// Evaluates `geo_expr` on the batch and boxes the resulting geos. `buffer`
// (Some for ST_DWithin) inflates every bbox by that distance.
pub(crate) fn bboxed_geo_batch(
    batch: RecordBatch,
    geo_expr: &Arc<dyn PhysicalExpr>,
    buffer: Option<f64>,
) -> Result<BBoxedGeoBatch> {
    let field = geo_expr.return_field(&batch.schema())?;
    let wkb = geo_expr.evaluate(&batch)?.into_array(batch.num_rows())?;
    let (geo_array, rects, geo_ids) = RawGeo::try_new(&field, wkb)?
        .with_buffer(buffer)
        .into_bboxes()?;
    Ok(BBoxedGeoBatch {
        batch,
        geo_array,
        rects,
        geo_ids,
    })
}

pub(crate) struct BBoxedGeoStream {
    inner: SendableRecordBatchStream,
    geo_expr: Arc<dyn PhysicalExpr>,
    buffer: Option<f64>,
}

impl BBoxedGeoStream {
    pub(crate) fn new(
        inner: SendableRecordBatchStream,
        geo_expr: Arc<dyn PhysicalExpr>,
        buffer: Option<f64>,
    ) -> Self {
        Self {
            inner,
            geo_expr,
            buffer,
        }
    }
}

impl Stream for BBoxedGeoStream {
    type Item = Result<BBoxedGeoBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                Poll::Ready(Some(bboxed_geo_batch(batch, &this.geo_expr, this.buffer)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod bbox_geo_batch_tests {
    use std::sync::Arc;

    use arrow_array::{Array, BinaryArray, Float64Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::PhysicalExpr;
    use geo_types::{Coord, Rect};

    use super::*;
    use crate::geospatial::test_utils::wkt_to_wkb;
    use crate::geospatial::udfs::udf_helpers::{geometry_field, GeoMetadata};

    #[test]
    fn test_boxed_geo_batch_squares() {
        let squares = [
            "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON((2 0, 3 0, 3 1, 2 1, 2 0))",
            "POLYGON((0 2, 1 2, 1 3, 0 3, 0 2))",
            "POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))",
        ];
        let wkbs: Vec<Vec<u8>> = squares.iter().map(|wkt| wkt_to_wkb(wkt)).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value_f64", DataType::Float64, false),
            Field::new("value_i64", DataType::Int64, false),
            geometry_field("geometry", DataType::Binary, false, &GeoMetadata::default())
                .as_ref()
                .clone(),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Int64Array::from(vec![10i64, 20, 30, 40])),
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|b| b.as_slice()),
                )),
            ],
        )
        .unwrap();

        let geo_expr = Arc::new(Column::new("geometry", 2)) as Arc<dyn PhysicalExpr>;
        let boxed = bboxed_geo_batch(batch, &geo_expr, None).unwrap();

        assert_eq!(boxed.geo_ids, vec![0, 1, 2, 3]);
        assert_eq!(
            boxed.rects[0],
            Rect::new(Coord { x: 0f32, y: 0f32 }, Coord { x: 1f32, y: 1f32 })
        );
        assert_eq!(
            boxed.rects[1],
            Rect::new(Coord { x: 2f32, y: 0f32 }, Coord { x: 3f32, y: 1f32 })
        );
        assert_eq!(
            boxed.rects[2],
            Rect::new(Coord { x: 0f32, y: 2f32 }, Coord { x: 1f32, y: 3f32 })
        );
        assert_eq!(
            boxed.rects[3],
            Rect::new(Coord { x: 2f32, y: 2f32 }, Coord { x: 3f32, y: 3f32 })
        );

        let f64_col = boxed
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(f64_col.values(), &[1.0, 2.0, 3.0, 4.0]);

        let i64_col = boxed
            .batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(i64_col.values(), &[10i64, 20, 30, 40]);
    }
}
