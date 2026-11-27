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

use arrow::array::AsArray;
use arrow::compute::{cast, max, min};
use arrow::datatypes::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, FieldRef, Schema};
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::{CastExpr, Column, Literal, TryCastExpr};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_plan::PhysicalExpr;
use geo_types::Rect;

use crate::geospatial::common::constants::WKB_DATA_TYPES;
use crate::geospatial::common::crs::geography_crs;
use crate::geospatial::geos::raw_geo::{GeoType, RawGeo};
use crate::geospatial::geos::IndexedBuildSide;

//***********************************************************************
// A physical expression that prunes probe-side chunks against the build-side
// spatial index.
// It holds the probe-side geometry expression (e.g. `st_point(lon, lat)`) as
// its single child — so callers know which probe columns to materialize — and
// the already-resolved build-side index.
//***********************************************************************

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
}

//***********************************************************************
// Various functions to handle bboxes as they traverse the physical expr.
//***********************************************************************

// The [min, max] of numeric values, as a 2 row batch with the field of
// the expression the values come from. nulls are ignored, they're the padding
// of the shorter columns. None if there are no finite values.
fn min_max_batch(values: &ArrayRef, field: &FieldRef) -> Result<Option<RecordBatch>> {
    let as_f64 = cast(values, &DataType::Float64)?;
    let as_f64 = as_f64.as_primitive::<Float64Type>();
    let (Some(min), Some(max)) = (min(as_f64), max(as_f64)) else {
        return Ok(None);
    };
    if !min.is_finite() || !max.is_finite() {
        return Ok(None);
    }

    let bounds: ArrayRef = Arc::new(Float64Array::from(vec![min, max]));
    let bounds = cast(&bounds, field.data_type())?;
    let schema = Arc::new(Schema::new(vec![Arc::clone(field)]));
    Ok(Some(RecordBatch::try_new(schema, vec![bounds])?))
}

// The [min, max] of a numeric expression, evaluated on all the values of the
// single column it references, so it can be anything, e.g. a CASE that isn't
// monotonic. None if it references more than one column.
fn numeric_node_batch(
    node: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
) -> Result<Option<RecordBatch>> {
    if collect_columns(node).len() != 1 {
        return Ok(None);
    }
    let field = node.return_field(&batch.schema())?;
    let values = node.evaluate(batch)?.into_array(batch.num_rows())?;
    min_max_batch(&values, &field)
}

// Produces the bbox of geo values, e.g. a geo column or a geo node's output, as a
// `bbox_of_rect` with `field`, the field of the expression the values come from.
// None if there are no geos or the bbox isn't valid.
fn bbox_of_geos(values: &ArrayRef, field: &FieldRef) -> Result<Option<RecordBatch>> {
    match RawGeo::try_new(field, Arc::clone(values))?.overall_bbox_f64()? {
        Some(bbox) => bbox_of_rect(field, &bbox),
        None => Ok(None),
    }
}

// A bbox as a batch of the `RawGeo::from_bbox` segments, cast back to the type
// of `field`, so the next node gets the type it expects. None if the bbox
// isn't valid.
fn bbox_of_rect(field: &FieldRef, bbox: &Rect<f64>) -> Result<Option<RecordBatch>> {
    // called by path since `min`/`max` are also Ord methods.
    let (min, max) = (Rect::min(*bbox), Rect::max(*bbox));
    RawGeo::from_bbox(field, min.x, min.y, max.x, max.y)?
        .map(|geos| with_wkb_field(geos.into_record_batch(field.name())?, field))
        .transpose()
}

// Gives a single column wkb batch the storage type and field of `field`, so
// the next node gets the type it expects. a `field` that isn't a wkb type,
// e.g. a geoarrow.polygon column, can't be cast to, the batch is kept as is.
fn with_wkb_field(batch: RecordBatch, field: &FieldRef) -> Result<RecordBatch> {
    if !WKB_DATA_TYPES.contains(field.data_type()) {
        return Ok(batch);
    }
    let column = cast(batch.column(0), field.data_type())?;
    let schema = Arc::new(Schema::new(vec![Arc::clone(field)]));
    Ok(RecordBatch::try_new(schema, vec![column])?)
}

// Rebinds a physical expr node to the bounds of its physical expr children:
// a single child column batch becomes a column of the returned batch, and
// the parent node reads it instead of the child. a child without bounds,
// e.g. a literal, is kept as is.
fn rebind_children(
    node: &Arc<dyn PhysicalExpr>,
    child_bounds: Vec<Option<RecordBatch>>,
) -> Result<(Arc<dyn PhysicalExpr>, RecordBatch)> {
    let num_rows = child_bounds
        .iter()
        .flatten()
        .next()
        .map_or(1, RecordBatch::num_rows);
    let mut children = Vec::with_capacity(child_bounds.len());
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (child, bounds) in node.children().into_iter().zip(child_bounds) {
        match bounds {
            Some(bounds) => {
                let name = format!("child_{}", columns.len());
                let field = bounds.schema().field(0).clone().with_name(&name);
                children.push(Arc::new(Column::new(&name, columns.len())) as Arc<dyn PhysicalExpr>);
                fields.push(field);
                columns.push(Arc::clone(bounds.column(0)));
            }
            None => children.push(Arc::clone(child)),
        }
    }

    let node = Arc::clone(node).with_new_children(children)?;
    let schema = Arc::new(Schema::new(fields));
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    let batch = RecordBatch::try_new_with_options(schema, columns, &options)?;
    Ok((node, batch))
}

// Whether a geo node's bbox is the bbox of the node evaluated on its
// children's bounds. a udf's arguments other than its geo input must be
// literals, except for st_point, whose inputs are all bounded.
fn is_supported(node: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
    if node.downcast_ref::<CastExpr>().is_some() || node.downcast_ref::<TryCastExpr>().is_some() {
        return Ok(true);
    }
    let Some(f) = node.downcast_ref::<ScalarFunctionExpr>() else {
        return Ok(false);
    };
    let name = f.fun().name();
    if name.eq_ignore_ascii_case("st_point") {
        return Ok(true);
    }
    let args = f.args();
    if !args
        .iter()
        .skip(1)
        .all(|arg| arg.downcast_ref::<Literal>().is_some())
    {
        return Ok(false);
    }

    if name.eq_ignore_ascii_case("st_setcrs") || name.eq_ignore_ascii_case("st_buffer") {
        Ok(true)
    } else if name.eq_ignore_ascii_case("st_togeography") {
        // only a relabel, from EPSG:4326 or a geography.
        let input = GeoType::from_field(name, args[0].return_field(schema)?.as_ref())?;
        Ok(input.is_geography() || input.crs() == Some(&geography_crs()))
    } else {
        Ok(false)
    }
}

// The bounds of a node over the values of `batch`: the [min, max] of a numeric
// node as a 2 row batch, the `bbox_of_rect` of a geo node.
// None if it can't be bounded. the nodes are bounded from the innermost out,
// each evaluated on its children's bounds, except numeric subtrees, which see
// all the values of their column.
fn bounds(node: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> Result<Option<RecordBatch>> {
    let schema = batch.schema();
    let field = node.return_field(&schema)?;
    if !GeoType::is_geo_field(&field) {
        return if field.data_type().is_numeric() {
            numeric_node_batch(node, batch)
        } else {
            Ok(None)
        };
    }
    if node.downcast_ref::<Column>().is_some() {
        let values = node.evaluate(batch)?.into_array(batch.num_rows())?;
        return bbox_of_geos(&values, &field);
    }
    if !is_supported(node, &schema)? {
        return Ok(None);
    }

    let mut child_bounds = Vec::new();
    for child in node.children() {
        if child.downcast_ref::<Literal>().is_some() {
            child_bounds.push(None);
            continue;
        }
        let Some(bounds) = bounds(child, batch)? else {
            return Ok(None);
        };
        child_bounds.push(Some(bounds));
    }
    let (node, batch) = rebind_children(node, child_bounds)?;
    let field = node.return_field(&batch.schema())?;
    let values = node.evaluate(&batch)?.into_array(batch.num_rows())?;
    bbox_of_geos(&values, &field)
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

    // One keep/skip verdict for the whole chunk. the batch has the values of the
    // columns the probe expression reads, the shorter columns padded with nulls,
    // it doesn't need to pair them, e.g. the chunk's 1D coordinates. anything
    // that can't be bounded keeps the chunk, this can be evaluated by others than
    // the readers. the buffer (Some for ST_DWithin) inflates the chunk's bbox so
    // it keeps the chunks within the distance of a build geometry.
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let bbox = match bounds(&self.probe_expr, batch)? {
            Some(corners) => {
                RawGeo::try_new(corners.schema().field(0), Arc::clone(corners.column(0)))?
                    .with_bbox_margin(self.probe_side_buffer)
                    .overall_bbox()?
            }
            None => None,
        };
        let keep = match bbox {
            Some(bbox) => {
                let (matches, _) = self.index.traverse(vec![bbox], vec![0]);
                !matches.is_empty()
            }
            None => true,
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

#[cfg(test)]
mod pruning_tests {
    use arrow_array::BinaryArray;
    use arrow_schema::Field;
    use datafusion::common::DFSchema;
    use serde_json::Value;

    use super::*;
    use crate::geospatial::geos::indexed_build_side::test_helpers::make_typed_indexed_build_side;
    use crate::geospatial::geos::joinable_geo::{GeoKernel, JoinableGeo, Planar, Spherical};
    use crate::geospatial::geos::raw_geo::GeoKind;
    use crate::geospatial::join_predicates::st_intersects::st_intersects;
    use crate::geospatial::test_utils::wkt_to_wkb;
    use crate::zarr_session_context;

    fn geometry_4326() -> GeoType {
        GeoType::geometry(Some(Value::String("EPSG:4326".into())))
    }

    fn geography() -> GeoType {
        geometry_4326().to_geography("test").unwrap()
    }

    fn probe_expr(sql: &str, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        let ctx = zarr_session_context().unwrap();
        let df_schema = DFSchema::try_from(schema.clone()).unwrap();
        let expr = ctx.parse_sql_expr(sql, &df_schema).unwrap();
        ctx.create_physical_expr(expr, &df_schema).unwrap()
    }

    fn any_intersects<K: GeoKernel>(geos: &ArrayRef, left: &[u8]) -> bool {
        let left = JoinableGeo::<K>::from_wkb(left).unwrap();
        geos.as_binary::<i32>()
            .iter()
            .flatten()
            .any(|bytes| st_intersects(&JoinableGeo::<K>::from_wkb(bytes).unwrap(), &left))
    }

    // Whether the probe expression, on the rows of `truth_batch`, intersects
    // `left`, and whether the pruning expression, built from an index of `left`,
    // keeps `pruning_batch`. a chunk with an intersection must be kept.
    fn truth_and_keep(
        sql: &str,
        left_wkt: &str,
        left_type: &GeoType,
        truth_batch: &RecordBatch,
        pruning_batch: &RecordBatch,
    ) -> (bool, bool) {
        let schema = truth_batch.schema();
        let expr = probe_expr(sql, &schema);
        let field = expr.return_field(&schema).unwrap();
        let geos = expr
            .evaluate(truth_batch)
            .unwrap()
            .into_array(truth_batch.num_rows())
            .unwrap();
        let geos = cast(&geos, &DataType::Binary).unwrap();
        let left = wkt_to_wkb(left_wkt);
        let truth = match GeoType::from_field("test", field.as_ref()).unwrap().kind() {
            GeoKind::Geometry => any_intersects::<Planar>(&geos, &left),
            GeoKind::Geography => any_intersects::<Spherical>(&geos, &left),
        };

        let index = make_typed_indexed_build_side(&[left_wkt], left_type);
        let pruning = ProbePruningExpr::new(expr, index, None);
        let keep = match pruning.evaluate(pruning_batch).unwrap() {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(keep))) => keep,
            other => panic!("expected a boolean scalar, got {other:?}"),
        };
        (truth, keep)
    }

    // The probe side is points, the same batch is used for both, as the zarr
    // reader does.
    fn points_truth_and_keep(
        sql: &str,
        left_wkt: &str,
        left_type: &GeoType,
        points: &[(f64, f64)],
    ) -> (bool, bool) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("lon", DataType::Float64, true),
            Field::new("lat", DataType::Float64, true),
        ]));
        let lons = Float64Array::from_iter_values(points.iter().map(|p| p.0));
        let lats = Float64Array::from_iter_values(points.iter().map(|p| p.1));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(lons), Arc::new(lats)]).unwrap();
        truth_and_keep(sql, left_wkt, left_type, &batch, &batch)
    }

    // The probe side is a single geo of type `right_type`, pruned from its
    // bbox, as the geoparquet reader does with its statistics.
    fn geo_truth_and_keep(
        sql: &str,
        left_wkt: &str,
        left_type: &GeoType,
        right_wkt: &str,
        right_type: &GeoType,
    ) -> (bool, bool) {
        let field = right_type.to_field("geom", DataType::Binary, true);
        let schema = Arc::new(Schema::new(vec![Arc::clone(&field)]));
        let wkb: ArrayRef = Arc::new(BinaryArray::from_iter_values([wkt_to_wkb(right_wkt)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&wkb)]).unwrap();

        let bbox = RawGeo::try_new(field.as_ref(), wkb)
            .unwrap()
            .overall_bbox_f64()
            .unwrap()
            .unwrap();
        let bounds = bbox_of_rect(&field, &bbox).unwrap().unwrap();
        truth_and_keep(sql, left_wkt, left_type, &batch, &bounds)
    }

    // the square's top edge, from (-25 85) to (25 85), peaks at ~85.47 as a
    // geography, past the vertices' latitude.
    #[test]
    fn points_relabelled_buffer_keeps_bulge() {
        let (truth, keep) = points_truth_and_keep(
            "st_togeography(st_buffer(st_point(lon, lat, 4326), 25.0, 'endcap=square'))",
            "POLYGON((-0.5 85.2, 0.5 85.2, 0.5 85.4, -0.5 85.4, -0.5 85.2))",
            &geography(),
            &[(0.0, 60.0)],
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn points_geography_buffer_prunes_far() {
        let (truth, keep) = points_truth_and_keep(
            "st_buffer(st_togeography(st_point(lon, lat, 4326)), 100000.0)",
            "POLYGON((99 -41, 101 -41, 101 -39, 99 -39, 99 -41))",
            &geography(),
            &[(0.0, 60.0)],
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }

    // the top edge, from (60 60) to (-60 60), peaks at ~73.9 as a geography,
    // and the bottom one at ~67.2, so the polygon covers (0 70).
    #[test]
    fn polygon_relabel_keeps_bulge() {
        let (truth, keep) = geo_truth_and_keep(
            "st_togeography(geom)",
            "POLYGON((-1 69.5, 1 69.5, 1 70.5, -1 70.5, -1 69.5))",
            &geography(),
            "POLYGON((-60 50, 60 50, 60 60, -60 60, -60 50))",
            &geometry_4326(),
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn polygon_geography_buffer_prunes_far() {
        let (truth, keep) = geo_truth_and_keep(
            "st_buffer(st_togeography(geom), 100000.0)",
            "POLYGON((149 -41, 151 -41, 151 -39, 149 -39, 149 -41))",
            &geography(),
            "POLYGON((-60 50, 60 50, 60 60, -60 60, -60 50))",
            &geometry_4326(),
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }

    const GEOMETRY_LINE: &str = "LINESTRING(-10 0, 10 0)";
    const GEOMETRY_POLYGON: &str = "POLYGON((-10 -10, 10 -10, 10 10, -10 10, -10 -10))";
    const NEAR_ORIGIN: &str = "POLYGON((-1 -1, 1 -1, 1 1, -1 1, -1 -1))";
    const FAR_FROM_ORIGIN: &str = "POLYGON((49 49, 51 49, 51 51, 49 51, 49 49))";

    // the line peaks at ~73.9 as a geography, past its vertices' latitude.
    const GEOGRAPHY_LINE: &str = "LINESTRING(-60 60, 60 60)";
    // the top edge peaks at ~73.9 as a geography, and the bottom one at ~67.2.
    const GEOGRAPHY_POLYGON: &str = "POLYGON((-60 50, 60 50, 60 60, -60 60, -60 50))";
    const FAR_FROM_BAND: &str = "POLYGON((149 -41, 151 -41, 151 -39, 149 -39, 149 -41))";

    #[test]
    fn geometry_line_keeps_near() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            NEAR_ORIGIN,
            &geometry_4326(),
            GEOMETRY_LINE,
            &geometry_4326(),
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn geometry_line_prunes_far() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            FAR_FROM_ORIGIN,
            &geometry_4326(),
            GEOMETRY_LINE,
            &geometry_4326(),
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }

    #[test]
    fn geometry_polygon_keeps_near() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            NEAR_ORIGIN,
            &geometry_4326(),
            GEOMETRY_POLYGON,
            &geometry_4326(),
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn geometry_polygon_prunes_far() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            FAR_FROM_ORIGIN,
            &geometry_4326(),
            GEOMETRY_POLYGON,
            &geometry_4326(),
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }

    #[test]
    fn geography_line_keeps_bulge() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            "POLYGON((-1 73.5, 1 73.5, 1 74.5, -1 74.5, -1 73.5))",
            &geography(),
            GEOGRAPHY_LINE,
            &geography(),
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn geography_line_prunes_far() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            FAR_FROM_BAND,
            &geography(),
            GEOGRAPHY_LINE,
            &geography(),
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }

    #[test]
    fn geography_polygon_keeps_bulge() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            "POLYGON((-1 69.5, 1 69.5, 1 70.5, -1 70.5, -1 69.5))",
            &geography(),
            GEOGRAPHY_POLYGON,
            &geography(),
        );
        assert!(truth, "the case should intersect");
        assert!(keep, "an intersecting chunk was pruned");
    }

    #[test]
    fn geography_polygon_prunes_far() {
        let (truth, keep) = geo_truth_and_keep(
            "geom",
            FAR_FROM_BAND,
            &geography(),
            GEOGRAPHY_POLYGON,
            &geography(),
        );
        assert!(!truth, "the case shouldn't intersect");
        assert!(!keep, "a far chunk was kept");
    }
}
