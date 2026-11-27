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

use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::builder::{BinaryBuilder, Float64Builder};
use arrow_array::{Array, ArrayRef, BinaryArray, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use datafusion::common::{exec_datafusion_err, exec_err, plan_err, Result};
use datafusion::logical_expr::ColumnarValue;
use geo_traits::to_geo::ToGeoGeometry;
use geo_traits::{
    CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use geo_types::{Coord, Rect};
use proj4rs::errors::Error as ProjError;
use proj4rs::transform::Transform;
use proj4rs::Proj;
use s2::latlng::LatLng;
use s2::point::{point_area, Point};
use s2::predicates::{robust_sign, Direction};
use s2::rect::Rect as S2Rect;
use s2::rect_bounder::RectBounder;
use serde_json::Value;
use wkb::reader::Wkb;
use wkb::writer::{write_geometry, WriteOptions};

use super::buffer::{geography_buffer, geometry_buffer, BufferStyle};
use super::joinable_geo::{JoinableGeo, Spherical};
use crate::geospatial::common::constants::{
    geo_err, EARTH_RADIUS_METERS, GEOGRAPHY_CRS, GEOGRAPHY_EDGES, LON_RANGE, WKB_DATA_TYPES,
};
use crate::geospatial::common::crs::{
    clamp_latitude, geography_crs, is_latitude_valid, normalized_crs, proj_from_crs,
    shift_longitude, transform_point, wrap_lon_diff,
};
use crate::geospatial::common::validation::{is_geometry_field, require_geometry};
use crate::geospatial::common::wkb::{
    geometry_field, line_wkb, point_at, point_wkb, wkb_at, wkb_field_like, GeoMetadata,
};

//*******************************************************
// Convenience structs to handle parsing crs and geography vs geometry
//*******************************************************
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeoKind {
    Geometry,
    // any non-planar edges are treated as spherical.
    Geography,
}

impl fmt::Display for GeoKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeoKind::Geometry => write!(f, "geometry"),
            GeoKind::Geography => write!(f, "geography"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeoType {
    crs: Option<Value>, // Note: a geography is always in EPSG:4326.
    kind: GeoKind,
}

impl GeoType {
    pub(crate) fn geometry(crs: Option<Value>) -> Self {
        Self {
            crs: crs.as_ref().map(normalized_crs),
            kind: GeoKind::Geometry,
        }
    }

    pub(crate) fn is_geo_field(field: &Field) -> bool {
        is_geometry_field(field)
    }

    pub(crate) fn from_field(context: &str, field: &Field) -> Result<Self> {
        require_geometry(field)?;
        let metadata = GeoMetadata::from_field(field)?;
        let kind = if metadata.is_geography() {
            GeoKind::Geography
        } else {
            GeoKind::Geometry
        };
        let geo_type = Self {
            crs: metadata.crs.as_ref().map(normalized_crs),
            kind,
        };

        if geo_type.is_geography() && geo_type.crs.as_ref() != Some(&geography_crs()) {
            return plan_err!(
                "{context}: a geography must be in {GEOGRAPHY_CRS}, got {}",
                geo_type.crs_display()
            );
        }
        Ok(geo_type)
    }

    // The common type of two operands, which must both be geometries or both
    // geographies, with the same crs.
    pub(crate) fn from_fields(context: &str, left: &Field, right: &Field) -> Result<Self> {
        let left = Self::from_field(context, left)?;
        let right = Self::from_field(context, right)?;
        if left.kind != right.kind {
            return plan_err!(
                "{context}: can't combine a {} with a {}",
                left.kind(),
                right.kind()
            );
        }
        if left.crs != right.crs {
            return plan_err!(
                "{context}: the crs of both sides must match, got {} and {}",
                left.crs_display(),
                right.crs_display()
            );
        }
        Ok(left)
    }

    pub(crate) fn is_geography(&self) -> bool {
        self.kind == GeoKind::Geography
    }

    pub(crate) fn kind(&self) -> GeoKind {
        self.kind
    }

    pub(crate) fn crs(&self) -> Option<&Value> {
        self.crs.as_ref()
    }

    // An arrow field of this type, i.e. tagged as wkb with this crs/edges.
    pub(crate) fn to_field(&self, name: &str, data_type: DataType, nullable: bool) -> FieldRef {
        let metadata = GeoMetadata {
            crs: self.crs.clone(),
            edges: self.is_geography().then(|| GEOGRAPHY_EDGES.to_string()),
        };
        geometry_field(name, data_type, nullable, &metadata)
    }

    // TTransform to a GeoType with a different crs.
    pub(crate) fn transformed(&self, context: &str, to_crs: Value) -> Result<Self> {
        if self.is_geography() {
            return plan_err!("{context}: expected a geometry, got a geography");
        }
        self.proj(context)?;
        let to = Self::geometry(Some(to_crs));
        to.proj(context)?;
        Ok(to)
    }

    // The type of a geo converted to a geography, a geography stays as is.
    pub(crate) fn to_geography(&self, context: &str) -> Result<Self> {
        if self.is_geography() {
            return Ok(self.clone());
        }
        self.proj(context)?;
        Ok(Self {
            crs: Some(geography_crs()),
            kind: GeoKind::Geography,
        })
    }

    fn proj(&self, context: &str) -> Result<Proj> {
        match &self.crs {
            Some(crs) => proj_from_crs(context, crs),
            None => plan_err!(
                "{context}: the geometry has no crs, set one with st_setcrs or when creating it"
            ),
        }
    }

    fn crs_display(&self) -> String {
        self.crs
            .as_ref()
            .map_or_else(|| "no crs".to_string(), Value::to_string)
    }
}

//*******************************************************
// A convenience struct that handles all the conventions with
// geometry/geography transformations. Holds an array ref with
// one or more geo wkbs in it.
//*******************************************************
#[derive(Debug, Clone)]
pub(crate) struct RawGeo {
    geo_type: GeoType,
    wkb: ArrayRef,
    // inflates every bbox produced from it, without changing the geos, e.g. the
    // distance of ST_DWithin.
    bbox_margin: Option<f64>,
}

//*******************************************************
// Different interfaces to construct the RawGeo.
//*******************************************************
impl RawGeo {
    pub(crate) fn try_new(field: &Field, wkb: ArrayRef) -> Result<Self> {
        let geo_type = GeoType::from_field(field.name(), field)?;
        if !WKB_DATA_TYPES.contains(wkb.data_type()) {
            return exec_err!(
                "{}: expected a Binary or BinaryView wkb array, got {}",
                field.name(),
                wkb.data_type()
            );
        }
        Ok(Self {
            geo_type,
            wkb,
            bbox_margin: None,
        })
    }

    // A single geo, labelled with the crs/edges of `field`, which can have any
    // geo encoding, e.g. a geoarrow.polygon column.
    pub(crate) fn from_wkb_value(field: &Field, bytes: &[u8]) -> Result<Self> {
        let wkb: ArrayRef = Arc::new(BinaryArray::from_iter_values([bytes]));
        Self::try_new(wkb_field_like(field)?.as_ref(), wkb)
    }

    // A bbox, as the segments around it, a 2 point line per row, labelled w
    // with `field`.
    pub(crate) fn from_bbox(
        field: &Field,
        xmin: f64,
        ymin: f64,
        xmax: f64,
        ymax: f64,
    ) -> Result<Option<Self>> {
        let geo_type = GeoType::from_field(field.name(), wkb_field_like(field)?.as_ref())?;
        let finite = [xmin, ymin, xmax, ymax].iter().all(|v| v.is_finite());
        if !finite || ymin > ymax || (xmin > xmax && !geo_type.is_geography()) {
            return Ok(None);
        }

        // correct for lat beyond -90/90 degrees, can happen after buffering.
        let (ymin, ymax) = if geo_type.crs() == Some(&geography_crs()) {
            (clamp_latitude(ymin), clamp_latitude(ymax))
        } else {
            (ymin, ymax)
        };
        let (xmin, xmax) = collapse_rounded_extent(xmin, xmax);
        let (ymin, ymax) = collapse_rounded_extent(ymin, ymax);

        let mut builder = BinaryBuilder::new();
        // a bbox of zero width or height has zero length segments, which
        // aren't valid lines.
        for (start, end) in bbox_segments(geo_type.kind, xmin, ymin, xmax, ymax) {
            if start == end {
                builder.append_value(point_wkb(start.0, start.1)?);
            } else {
                builder.append_value(line_wkb(&[start, end])?);
            }
        }
        Ok(Some(Self::from_built(builder, geo_type)))
    }

    // A column of points, None for a null row.
    pub(crate) fn from_points(
        rows: impl IntoIterator<Item = Option<(f64, f64)>>,
        crs: Option<Value>,
    ) -> Result<Self> {
        let mut builder = BinaryBuilder::new();
        for row in rows {
            match row {
                Some((x, y)) => builder.append_value(point_wkb(x, y)?),
                None => builder.append_null(),
            }
        }
        Ok(Self::from_built(builder, GeoType::geometry(crs)))
    }

    // A column of 2 point line strings, from 2 columns of points of the type of
    // `field`. a row is null if either point is null or empty.
    pub(crate) fn line_from_points(
        field: &Field,
        starts: &ArrayRef,
        ends: &ArrayRef,
    ) -> Result<Self> {
        let geo_type = GeoType::from_field(field.name(), field)?;
        let mut builder = BinaryBuilder::new();
        for row in 0..starts.len() {
            match (point_at(starts, row)?, point_at(ends, row)?) {
                (Some(start), Some(end)) => builder.append_value(line_wkb(&[start, end])?),
                _ => builder.append_null(),
            }
        }
        Ok(Self::from_built(builder, geo_type))
    }

    fn from_built(mut builder: BinaryBuilder, geo_type: GeoType) -> Self {
        Self {
            geo_type,
            wkb: Arc::new(builder.finish()),
            bbox_margin: None,
        }
    }
}

//*******************************************************
// Core bounding box logic.
//*******************************************************
impl RawGeo {
    fn row_bbox(&self, row: usize) -> Result<Option<Rect<f32>>> {
        Ok(self
            .unbuffered_row_bbox(row)?
            .map(|rect| inflate(rect, self.bbox_margin, self.geo_type.is_geography())))
    }

    fn unbuffered_row_bbox(&self, row: usize) -> Result<Option<Rect<f32>>> {
        let Some(bytes) = wkb_at(&self.wkb, row) else {
            return Ok(None);
        };
        let wkb = Wkb::try_new(bytes).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
        Ok(match self.geo_type.kind {
            GeoKind::Geometry => bbox_of(&wkb),
            GeoKind::Geography if has_only_trivial_rings(&wkb) => geography_bbox_of(&wkb),
            GeoKind::Geography => JoinableGeo::<Spherical>::from_wkb(bytes)
                .map_err(geo_err)?
                .bounding_box()
                .map(|b| outward_f32_rect(b.min_x, b.min_y, b.max_x, b.max_y)),
        })
    }
}
// The bounding box of a geography, accounting for its geodesic edges, which
// can bulge past their vertices towards the poles. a bbox crossing the antimeridian
// gets the full longitude range.
// Note: this more or less duplicates some logic from JoinableGeo, which we will
// tolerate here because it's skipping some things that are not necessary that
// JoinableGeo does.
fn geography_bbox_of(wkb: &Wkb) -> Option<Rect<f32>> {
    let mut bound = S2Rect::empty();
    let mut add_chain = |coords: &mut dyn Iterator<Item = (f64, f64)>| {
        let mut bounder = RectBounder::new();
        for (x, y) in coords {
            bounder.add_point(&Point::from(LatLng::from_degrees(y, x)));
        }
        bound = bound.union(&bounder.get_bound());
    };
    match wkb.as_type() {
        GeometryType::Point(p) => {
            if let Some(c) = p.coord() {
                add_chain(&mut std::iter::once((c.x(), c.y())));
            }
        }
        GeometryType::MultiPoint(mp) => {
            for p in mp.points() {
                if let Some(c) = p.coord() {
                    add_chain(&mut std::iter::once((c.x(), c.y())));
                }
            }
        }
        GeometryType::LineString(ls) => add_chain(&mut ls.coords().map(|c| (c.x(), c.y()))),
        GeometryType::MultiLineString(mls) => {
            for ls in mls.line_strings() {
                add_chain(&mut ls.coords().map(|c| (c.x(), c.y())));
            }
        }
        GeometryType::Polygon(poly) => {
            if let Some(ext) = poly.exterior() {
                add_chain(&mut ext.coords().map(|c| (c.x(), c.y())));
            }
        }
        GeometryType::MultiPolygon(mp) => {
            for poly in mp.polygons() {
                if let Some(ext) = poly.exterior() {
                    add_chain(&mut ext.coords().map(|c| (c.x(), c.y())));
                }
            }
        }
        _ => return None,
    }

    if bound.is_empty() {
        return None;
    }
    let (lon_lo, lon_hi) = if bound.is_inverted() {
        LON_RANGE
    } else {
        (bound.lng_lo().deg(), bound.lng_hi().deg())
    };
    Some(outward_f32_rect(
        lon_lo,
        bound.lat_lo().deg(),
        lon_hi,
        bound.lat_hi().deg(),
    ))
}

// Computes the bounding box of each geometry.
fn bbox_of(wkb: &Wkb) -> Option<Rect<f32>> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    let mut add = |x: f64, y: f64| {
        let x = x as f32;
        let y = y as f32;
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    };

    match wkb.as_type() {
        GeometryType::Point(p) => {
            if let Some(c) = p.coord() {
                add(c.x(), c.y());
            }
        }
        GeometryType::MultiPoint(mp) => {
            for p in mp.points() {
                if let Some(c) = p.coord() {
                    add(c.x(), c.y());
                }
            }
        }
        GeometryType::LineString(ls) => {
            for c in ls.coords() {
                add(c.x(), c.y());
            }
        }
        GeometryType::MultiLineString(mls) => {
            for ls in mls.line_strings() {
                for c in ls.coords() {
                    add(c.x(), c.y());
                }
            }
        }
        GeometryType::Polygon(poly) => {
            if let Some(ext) = poly.exterior() {
                for c in ext.coords() {
                    add(c.x(), c.y());
                }
            }
        }
        GeometryType::MultiPolygon(mp) => {
            for poly in mp.polygons() {
                if let Some(ext) = poly.exterior() {
                    for c in ext.coords() {
                        add(c.x(), c.y());
                    }
                }
            }
        }
        _ => return None,
    }

    if min_x.is_finite() {
        Some(Rect::new(
            Coord { x: min_x, y: min_y },
            Coord { x: max_x, y: max_y },
        ))
    } else {
        None
    }
}

//*******************************************************
// Methods that extract some infromation from the geometries.
//*******************************************************
impl RawGeo {
    // The area of each geo, null for a null row. only polygons have an area, in
    // the crs units squared for a geometry, in square meters for a geography.
    pub(crate) fn area(&self) -> Result<ArrayRef> {
        let mut builder = Float64Builder::with_capacity(self.wkb.len());
        for row in 0..self.wkb.len() {
            let Some(bytes) = wkb_at(&self.wkb, row) else {
                builder.append_null();
                continue;
            };
            let wkb =
                Wkb::try_new(bytes).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
            builder.append_value(match self.geo_type.kind {
                GeoKind::Geometry => planar_area(&wkb),
                GeoKind::Geography => {
                    spherical_area(&wkb) * EARTH_RADIUS_METERS * EARTH_RADIUS_METERS
                }
            });
        }
        Ok(Arc::new(builder.finish()))
    }

    // A bbox covering every geo of the column, None if there are none.
    pub(crate) fn overall_bbox(&self) -> Result<Option<Rect<f32>>> {
        let mut bbox = None;
        for row in 0..self.wkb.len() {
            bbox = union_rects(bbox, self.unbuffered_row_bbox(row)?);
        }
        // inflated once, for a geography a row's inflation only accounts for
        // its own latitudes, not the other rows'.
        Ok(bbox.map(|bbox| inflate(bbox, self.bbox_margin, self.geo_type.is_geography())))
    }

    pub(crate) fn overall_bbox_f64(&self) -> Result<Option<Rect<f64>>> {
        Ok(self.overall_bbox()?.map(|rect| {
            Rect::new(
                Coord {
                    x: rect.min().x as f64,
                    y: rect.min().y as f64,
                },
                Coord {
                    x: rect.max().x as f64,
                    y: rect.max().y as f64,
                },
            )
        }))
    }
}

// Expands a rect by `margin` on all four sides, if any. for a geography the rect
// is in lon/lat degrees and the margin in meters.
fn inflate(rect: Rect<f32>, margin: Option<f64>, is_geography: bool) -> Rect<f32> {
    let Some(margin) = margin else {
        return rect;
    };
    if is_geography {
        return inflate_geography(rect, margin);
    }
    let b = margin as f32;
    Rect::new(
        Coord {
            x: rect.min().x - b,
            y: rect.min().y - b,
        },
        Coord {
            x: rect.max().x + b,
            y: rect.max().y + b,
        },
    )
}

// The longitude expansion is the widest one, at the latitude closest to a pole.
// past a pole, or if the expansion reaches all the longitudes, the longitude
// range is full.
fn inflate_geography(rect: Rect<f32>, margin: f64) -> Rect<f32> {
    let angle = margin / EARTH_RADIUS_METERS;
    let lat_lo = rect.min().y as f64 - angle.to_degrees();
    let lat_hi = rect.max().y as f64 + angle.to_degrees();

    let (mut lon_lo, mut lon_hi) = LON_RANGE;
    if is_latitude_valid(lat_lo) && is_latitude_valid(lat_hi) {
        let max_abs_lat = lat_lo.abs().max(lat_hi.abs()).to_radians();
        let sin_dlon = angle.sin() / max_abs_lat.cos();
        if angle < std::f64::consts::FRAC_PI_2 && sin_dlon < 1.0 {
            let dlon = sin_dlon.asin().to_degrees();
            lon_lo = rect.min().x as f64 - dlon;
            lon_hi = rect.max().x as f64 + dlon;
            if lon_lo < LON_RANGE.0 || lon_hi > LON_RANGE.1 {
                (lon_lo, lon_hi) = LON_RANGE;
            }
        }
    }

    outward_f32_rect(
        lon_lo,
        clamp_latitude(lat_lo),
        lon_hi,
        clamp_latitude(lat_hi),
    )
}

//*******************************************************
// Methods that consume the RawGeo and return a new RawGeo
//*******************************************************
impl RawGeo {
    // Wraps the longitudes to -180..180, vertex by vertex.
    pub(crate) fn shift_longitude(self) -> Result<Self> {
        let wkb = map_coordinates(&self.wkb, |x, y, z| Ok((shift_longitude(x), y, z)))?;
        Ok(Self {
            geo_type: self.geo_type,
            wkb,
            bbox_margin: None,
        })
    }

    // Reprojects to the crs of `to` and relabels as `to`, which comes from
    // `GeoType::transformed` or `GeoType::to_geography`.
    pub(crate) fn transform(self, to: GeoType) -> Result<Self> {
        let wkb = if self.geo_type.crs == to.crs && self.geo_type.kind == to.kind {
            cast(&self.wkb, &DataType::Binary)?
        } else if self.geo_type.crs == to.crs {
            map_coordinates(&self.wkb, to_geography_coords)?
        } else {
            let from_proj = self.geo_type.proj("transform")?;
            let to_proj = to.proj("transform")?;
            map_coordinates(&self.wkb, |x, y, z| {
                let (x, y, z) = transform_point(&from_proj, &to_proj, x, y, z)?;
                match to.kind {
                    GeoKind::Geometry => Ok((x, y, z)),
                    GeoKind::Geography => to_geography_coords(x, y, z),
                }
            })?
        };
        Ok(Self {
            geo_type: to,
            wkb,
            bbox_margin: None,
        })
    }

    // Buffers each geo by its row's radius, null if the geo, radius or style is
    // null. the radius is in the crs units for a geometry, in meters for a
    // geography.
    pub(crate) fn buffer(
        self,
        radius_at: impl Fn(usize) -> Option<f64>,
        style_at: impl Fn(usize) -> Result<Option<BufferStyle>>,
    ) -> Result<Self> {
        let mut buf = Vec::new();
        let mut builder = BinaryBuilder::with_capacity(self.wkb.len(), 0);
        for row in 0..self.wkb.len() {
            let (Some(bytes), Some(radius), Some(style)) =
                (wkb_at(&self.wkb, row), radius_at(row), style_at(row)?)
            else {
                builder.append_null();
                continue;
            };
            match self.geo_type.kind {
                GeoKind::Geometry => geometry_buffer(bytes, radius, &style, &mut buf)?,
                GeoKind::Geography => geography_buffer(bytes, radius, &style, &mut buf)?,
            }
            builder.append_value(&buf);
        }
        Ok(Self::from_built(builder, self.geo_type))
    }

    pub(crate) fn with_bbox_margin(mut self, bbox_margin: Option<f64>) -> Self {
        self.bbox_margin = bbox_margin;
        self
    }
}

//*******************************************************
// Methods that consume the RawGeo and return a Datafusion or
// arrow type.
//*******************************************************
impl RawGeo {
    #[cfg(test)]
    pub(crate) fn into_array(self) -> ArrayRef {
        self.wkb
    }

    // A single column batch of the geos, named `name`.
    pub(crate) fn into_record_batch(self, name: &str) -> Result<RecordBatch> {
        let field = self
            .geo_type
            .to_field(name, self.wkb.data_type().clone(), true);
        let schema = Arc::new(Schema::new(vec![field]));
        Ok(RecordBatch::try_new(schema, vec![self.wkb])?)
    }

    pub(crate) fn into_columnar_value(self) -> ColumnarValue {
        ColumnarValue::Array(self.wkb)
    }

    // The wkb array, with the bbox of each geo and its row. null and empty geos
    // are skipped.
    pub(crate) fn into_bboxes(self) -> Result<(ArrayRef, Vec<Rect<f32>>, Vec<usize>)> {
        let mut rects = Vec::new();
        let mut rows = Vec::new();
        for row in 0..self.wkb.len() {
            if let Some(rect) = self.row_bbox(row)? {
                rects.push(rect);
                rows.push(row);
            }
        }
        Ok((self.wkb, rects, rows))
    }
}

//*******************************************************
// Area calculation helpers.
//*******************************************************

// A polygon's area is its shell area minus the area of each hole. ring roles
// come from the wkb structure, not the winding, so each ring is measured by the
// absolute value of its shoelace sum.
fn planar_area(wkb: &Wkb) -> f64 {
    fn polygon(poly: &impl PolygonTrait<T = f64>) -> f64 {
        let shell = poly.exterior().map(|r| planar_ring_area(&r)).unwrap_or(0.0);
        let holes: f64 = poly.interiors().map(|h| planar_ring_area(&h)).sum();
        shell - holes
    }
    match wkb.as_type() {
        GeometryType::Polygon(poly) => polygon(poly),
        GeometryType::MultiPolygon(mp) => mp.polygons().map(|p| polygon(&p)).sum(),
        _ => 0.0,
    }
}

// Area of a single ring. The coordinates are shifted by the first vertex
// before the cross products to keep precision when the ring sits far from
// the origin.
fn planar_ring_area(ring: &impl LineStringTrait<T = f64>) -> f64 {
    let mut coords = ring.coords();
    let Some(origin) = coords.next() else {
        return 0.0;
    };
    let (ox, oy) = (origin.x(), origin.y());

    let mut sum = 0.0;
    let (mut px, mut py) = (0.0, 0.0);
    for c in coords {
        let (cx, cy) = (c.x() - ox, c.y() - oy);
        sum += px * cy - cx * py;
        (px, py) = (cx, cy);
    }
    (sum / 2.0).abs()
}

// The area on the unit sphere. a ring's interior is on its left, holes included,
// so a hole's area is the complement of its left side.
fn spherical_area(wkb: &Wkb) -> f64 {
    fn polygon(poly: &impl PolygonTrait<T = f64>) -> f64 {
        let shell = poly.exterior().map(|r| ring_left_area(&r)).unwrap_or(0.0);
        let holes: f64 = poly
            .interiors()
            .map(|h| 4.0 * PI - ring_left_area(&h))
            .sum();
        shell - holes
    }
    match wkb.as_type() {
        GeometryType::Polygon(poly) => polygon(poly),
        GeometryType::MultiPolygon(mp) => mp.polygons().map(|p| polygon(&p)).sum(),
        _ => 0.0,
    }
}

// The area on the left of a ring, on the unit sphere: the signed areas of the
// triangles fanning out from its first vertex sum to its signed area, up to a
// multiple of 4 pi, i.e. the whole sphere, so the sum is taken mod 4 pi.
fn ring_left_area(ring: &impl LineStringTrait<T = f64>) -> f64 {
    let mut vertices: Vec<Point> = ring
        .coords()
        .map(|c| Point::from(LatLng::from_degrees(c.y(), c.x())))
        .collect();
    vertices.pop();
    if vertices.len() < 3 {
        return 0.0;
    }

    let v0 = &vertices[0];
    let signed: f64 = vertices[1..]
        .windows(2)
        .map(|w| {
            let area = point_area(v0, &w[0], &w[1]);
            match robust_sign(v0, &w[0], &w[1]) {
                Direction::CounterClockwise => area,
                Direction::Clockwise => -area,
                Direction::Indeterminate => 0.0,
            }
        })
        .sum();
    signed.rem_euclid(4.0 * PI)
}

//*******************************************************
// Couple more local helpers
//*******************************************************
fn outward_f32_rect(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Rect<f32> {
    Rect::new(
        Coord {
            x: (xmin as f32).next_down(),
            y: (ymin as f32).next_down(),
        },
        Coord {
            x: (xmax as f32).next_up(),
            y: (ymax as f32).next_up(),
        },
    )
}

// A geography's coordinates: the longitude wrapped, like PostGIS, a latitude
// past a pole is rejected.
fn to_geography_coords(x: f64, y: f64, z: f64) -> std::result::Result<(f64, f64, f64), ProjError> {
    if !is_latitude_valid(y) {
        return Err(ProjError::LatitudeOutOfRange);
    }
    Ok((shift_longitude(x), y, z))
}

// Applies `f` to every vertex of every 2D geo, into a Binary array.
fn map_coordinates(
    wkb: &ArrayRef,
    mut f: impl FnMut(f64, f64, f64) -> std::result::Result<(f64, f64, f64), ProjError>,
) -> Result<ArrayRef> {
    let opts = WriteOptions::default();
    let mut buf = Vec::new();
    let mut builder = BinaryBuilder::with_capacity(wkb.len(), 0);
    for row in 0..wkb.len() {
        let Some(bytes) = wkb_at(wkb, row) else {
            builder.append_null();
            continue;
        };
        let geo = Wkb::try_new(bytes).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
        if geo.dim() != Dimensions::Xy {
            return exec_err!("only 2D geometries are supported");
        }

        // geo_types can't represent empty points, they're left as is.
        let mut geom = match geo.try_to_geometry() {
            Some(geom) => geom,
            None if matches!(geo.as_type(), GeometryType::Point(p) if p.coord().is_none()) => {
                builder.append_value(bytes);
                continue;
            }
            None => return exec_err!("unsupported geometry containing empty parts"),
        };
        geom.transform_coordinates(&mut f)
            .map_err(|e| exec_datafusion_err!("{e}"))?;

        buf.clear();
        write_geometry(&mut buf, &geom, &opts).map_err(|e| exec_datafusion_err!("{e}"))?;
        builder.append_value(&buf);
    }
    Ok(Arc::new(builder.finish()))
}

type Segment = ((f64, f64), (f64, f64));

// The segments around a bbox, from point to point of a path around it.
// An extent of at most the outward f32 rounding of a single value, e.g. the
// bbox of a point, is collapsed to that value, so that its segments aren't
// tiny lines, which some operations, e.g. a buffer, treat as degenerate.
fn collapse_rounded_extent(min: f64, max: f64) -> (f64, f64) {
    if min <= max && (min as f32).next_up().next_up() >= max as f32 {
        let mid = (min + max) / 2.0;
        (mid, mid)
    } else {
        (min, max)
    }
}

fn bbox_segments(kind: GeoKind, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<Segment> {
    let width = if xmin <= xmax {
        xmax - xmin
    } else {
        xmax + 360.0 - xmin
    };
    let path = match (kind, width >= 180.0) {
        (GeoKind::Geography, true) => return wide_geography_segments(ymin, ymax),
        (GeoKind::Geography, false) => vec![
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ],
        (GeoKind::Geometry, _) => geometry_box_ring(xmin, ymin, xmax, ymax),
    };
    path_segments(&path)
}

fn path_segments(path: &[(f64, f64)]) -> Vec<Segment> {
    path.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

// The closed path through a geometry bbox's corners, its horizontal edges split
// in steps of at most 90, so that once relabelled to a geography they still go
// the same way around.
fn geometry_box_ring(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<(f64, f64)> {
    let steps = ((xmax - xmin) / 90.0).ceil().max(1.0) as usize;
    let x_at = |i: usize| xmin + (xmax - xmin) * i as f64 / steps as f64;
    let bottom = (0..=steps).map(|i| (x_at(i), ymin));
    let top = (0..=steps).rev().map(|i| (x_at(i), ymax));
    bottom.chain(top).chain([(xmin, ymin)]).collect()
}

// A geography bbox 180+ degrees wide: edges between its points can go the
// other way around, and over a pole. bounded by all longitudes, up to the
// poles on the sides the bbox reaches.
fn wide_geography_segments(ymin: f64, ymax: f64) -> Vec<Segment> {
    let lat_lo = if ymin < 0.0 { -90.0 } else { ymin };
    let lat_hi = if ymax > 0.0 { 90.0 } else { ymax };

    // a parallel only bulges poleward, so it's kept off the poles.
    let parallel_lat = if lat_lo > -90.0 {
        lat_lo
    } else if lat_hi < 90.0 {
        lat_hi
    } else {
        0.0
    };
    let parallel: Vec<(f64, f64)> = [-180.0, -90.0, 0.0, 90.0, 180.0]
        .iter()
        .map(|&x| (x, parallel_lat))
        .collect();
    let meridian = [(0.0, lat_lo), (0.0, (lat_lo + lat_hi) / 2.0), (0.0, lat_hi)];

    let mut segments = path_segments(&parallel);
    segments.extend(path_segments(&meridian));
    segments
}

fn union_rects(a: Option<Rect<f32>>, b: Option<Rect<f32>>) -> Option<Rect<f32>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Rect::new(
            Coord {
                x: a.min().x.min(b.min().x),
                y: a.min().y.min(b.min().y),
            },
            Coord {
                x: a.max().x.max(b.max().x),
                y: a.max().y.max(b.max().y),
            },
        )),
        (a, b) => a.or(b),
    }
}

// Whether the geodesic edge bbox of a geography covers it, i.e. none of its
// polygons contains a pole, which is the case if every ring is trivial.
fn has_only_trivial_rings(wkb: &Wkb) -> bool {
    fn polygon(poly: &impl PolygonTrait<T = f64>) -> bool {
        poly.exterior().is_none_or(|ring| is_trivial_ring(&ring))
            && poly.interiors().all(|ring| is_trivial_ring(&ring))
    }
    match wkb.as_type() {
        GeometryType::Polygon(poly) => polygon(poly),
        GeometryType::MultiPolygon(mp) => mp.polygons().all(|poly| polygon(&poly)),
        _ => true,
    }
}

// A ring that doesn't touch a pole, doesn't wind around one, and is CCW, so it
// has neither pole on its left. a CW ring has both, its left side is the
// complement of the area it encloses.
fn is_trivial_ring(ring: &impl LineStringTrait<T = f64>) -> bool {
    const EPS: f64 = 1e-9;
    let mut coords = ring.coords().map(|c| (c.x(), c.y()));
    let Some((x0, y0)) = coords.next() else {
        return true;
    };

    let (mut prev_x, mut prev_y) = (x0, y0);
    let mut unwrapped_x = x0;
    let mut winding = 0.0;
    let mut area = 0.0;
    for (x, y) in coords {
        if prev_y.abs() >= 90.0 - EPS || y.abs() >= 90.0 - EPS {
            return false;
        }
        let d = wrap_lon_diff(x - prev_x);
        if d.abs() >= 180.0 - EPS {
            return false;
        }
        area += unwrapped_x * y - (unwrapped_x + d) * prev_y;
        unwrapped_x += d;
        winding += d;
        (prev_x, prev_y) = (x, y);
    }
    winding.abs() < EPS && area > 0.0
}
