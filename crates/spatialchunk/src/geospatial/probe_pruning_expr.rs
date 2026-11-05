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

use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::PhysicalExpr;
use geo_types::{Coord, Rect};

use crate::geospatial::geos::boxed_geo_batch::inflate;
use crate::geospatial::geos::{BBoxedGeoBatch, IndexedBuildSide};

/// A physical expression that prunes probe-side chunks against the build-side
/// spatial index.
///
/// It holds the probe-side geometry expression (e.g. `st_point(lon, lat)`) as
/// its single child — so callers know which probe columns to materialize — and
/// the already-resolved build-side index.
pub(crate) struct ProbePruningExpr {
    probe_expr: Arc<dyn PhysicalExpr>,
    index: Arc<IndexedBuildSide>,

    // For ST_DWithin: the distance threshold, used to inflate each probe chunk's
    // bbox before testing it against the build index.
    probe_side_buffer: Option<f64>,
}

impl ProbePruningExpr {
    pub(crate) fn new(
        probe_expr: Arc<dyn PhysicalExpr>,
        index: Arc<IndexedBuildSide>,
        probe_side_buffer: Option<f64>,
    ) -> Self {
        Self {
            probe_expr,
            index,
            probe_side_buffer,
        }
    }

    // The probe geometry column, if the probe expression is a bare column, e.g.
    // a geometry column read from a file, as opposed to st_point(lon, lat).
    pub(crate) fn probe_column(&self) -> Option<&str> {
        self.probe_expr.downcast_ref::<Column>().map(Column::name)
    }

    // Whether a probe bbox may intersect any build side geometry. used to prune
    // with bboxes that are already known, e.g. from parquet statistics, where
    // there's no record batch to evaluate. the buffer (Some for ST_DWithin)
    // inflates the bbox, like in `evaluate`.
    pub(crate) fn may_intersect(&self, rect: Rect<f64>) -> bool {
        let rect = Rect::new(
            Coord {
                x: (rect.min().x as f32).next_down(),
                y: (rect.min().y as f32).next_down(),
            },
            Coord {
                x: (rect.max().x as f32).next_up(),
                y: (rect.max().y as f32).next_up(),
            },
        );
        let rect = match self.probe_side_buffer {
            Some(buffer) => inflate(rect, buffer),
            None => rect,
        };
        let (matches, _) = self.index.traverse(vec![rect], vec![0]);
        !matches.is_empty()
    }
}

/// Unions a set of rects into a single enclosing bbox, or `None` if empty.
fn union_rects(rects: &[Rect<f32>]) -> Option<Rect<f32>> {
    rects.iter().copied().reduce(|a, b| {
        Rect::new(
            Coord {
                x: a.min().x.min(b.min().x),
                y: a.min().y.min(b.min().y),
            },
            Coord {
                x: a.max().x.max(b.max().x),
                y: a.max().y.max(b.max().y),
            },
        )
    })
}

// Identity is defined by the probe expression alone; the build-side future is
// not comparable/hashable and there is only ever one index per join.
impl PartialEq for ProbePruningExpr {
    fn eq(&self, other: &Self) -> bool {
        self.probe_expr.eq(&other.probe_expr)
    }
}

impl Eq for ProbePruningExpr {}

impl Hash for ProbePruningExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.probe_expr.hash(state);
    }
}

impl fmt::Debug for ProbePruningExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProbePruningExpr")
            .field("probe_expr", &self.probe_expr)
            .finish_non_exhaustive()
    }
}

impl Display for ProbePruningExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ProbePruningExpr({})", self.probe_expr)
    }
}

impl PhysicalExpr for ProbePruningExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        // One keep/skip verdict for the whole chunk. Callers pass either
        // the chunk's full coordinate columns or just its min/max corners. The
        // buffer (Some for ST_DWithin) inflates the probe bboxes so the coarse
        // filter keeps chunks within the distance of a build geometry.
        let geo_batch =
            BBoxedGeoBatch::new(batch.clone(), &self.probe_expr, self.probe_side_buffer)?;

        let keep = match union_rects(&geo_batch.rects) {
            // No geometry in the chunk → nothing can match → prune it.
            None => false,
            Some(chunk_rect) => {
                let (matches, _) = self.index.traverse(vec![chunk_rect], vec![0]);
                !matches.is_empty()
            }
        };

        Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(keep))))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.probe_expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(ProbePruningExpr::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.index),
            self.probe_side_buffer,
        )))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ProbePruningExpr({})", self.probe_expr)
    }
}
