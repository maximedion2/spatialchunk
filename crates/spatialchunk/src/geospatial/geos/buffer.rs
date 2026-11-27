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

use datafusion::common::{exec_datafusion_err, exec_err, Result};
use geo_traits::to_geo::ToGeoGeometry;
use geo_traits::{Dimensions, GeometryTrait};
use geo_types::{Coord, Geometry, LineString, MultiPolygon, Polygon};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::mesh::float::outline::offset::OutlineOffset;
use i_overlay::mesh::float::stroke::offset::StrokeOffset;
use i_overlay::mesh::float::style::{LineCap, LineJoin, OutlineStyle, StrokeStyle};
use proj4rs::errors::Error as ProjError;
use proj4rs::transform::Transform;
use proj4rs::Proj;
use s2::edgeutil::interpolate;
use s2::latlng::LatLng;
use s2::point::Point;
use s2::r3::vector::Vector;
use wkb::reader::Wkb;
use wkb::writer::{write_geometry, write_multi_polygon, write_polygon, WriteOptions};

use super::joinable_geo::{JoinableGeo, Planar};
use crate::geospatial::common::constants::{geo_err, EARTH_RADIUS_METERS};
use crate::geospatial::common::crs::transform_point;

// Default number of segments per quarter circle (matches PostGIS).
pub const DEFAULT_QUAD_SEGS: usize = 8;

// Default mitre limit (matches PostGIS).
pub const DEFAULT_MITRE_LIMIT: f64 = 5.0;

//************************************************
// Enums for the buffer style and its components, the join style
// and the endcap.
//************************************************
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct BufferStyle {
    pub(crate) quad_segs: usize,
    pub(crate) endcap: EndCap,
    pub(crate) join: Join,
}

impl Default for BufferStyle {
    fn default() -> Self {
        BufferStyle {
            quad_segs: DEFAULT_QUAD_SEGS,
            endcap: EndCap::default(),
            join: Join::default(),
        }
    }
}

impl BufferStyle {
    // Parses a PostGIS style string: space-separated `key=value` pairs, e.g.
    // 'quad_segs=4 endcap=flat join=mitre mitre_limit=2'. Unset keys keep
    // their defaults. `mitre_limit` only matters with `join=mitre`, so it is
    // collected separately and applied once all pairs are read.
    pub(crate) fn parse(text: &str) -> Result<BufferStyle> {
        let mut style = BufferStyle::default();
        let mut mitre_limit = DEFAULT_MITRE_LIMIT;
        for pair in text.split_whitespace() {
            let Some((key, value)) = pair.split_once('=') else {
                return exec_err!("st_buffer: invalid style parameter '{pair}'");
            };
            let value = value.to_ascii_lowercase();
            match key.to_ascii_lowercase().as_str() {
                "quad_segs" => {
                    style.quad_segs =
                        value
                            .parse::<usize>()
                            .ok()
                            .filter(|&q| q > 0)
                            .ok_or_else(|| {
                                exec_datafusion_err!("st_buffer: invalid quad_segs '{value}'")
                            })?;
                }
                "endcap" => {
                    style.endcap = match value.as_str() {
                        "round" => EndCap::Round,
                        "flat" | "butt" => EndCap::Flat,
                        "square" => EndCap::Square,
                        _ => return exec_err!("st_buffer: invalid endcap '{value}'"),
                    };
                }
                "join" => {
                    style.join = match value.as_str() {
                        "round" => Join::Round,
                        "mitre" | "miter" => Join::Mitre {
                            limit: DEFAULT_MITRE_LIMIT,
                        },
                        "bevel" => Join::Bevel,
                        _ => return exec_err!("st_buffer: invalid join '{value}'"),
                    };
                }
                "mitre_limit" | "miter_limit" => {
                    mitre_limit = value
                        .parse::<f64>()
                        .ok()
                        .filter(|l| l.is_finite() && *l >= 0.0)
                        .ok_or_else(|| {
                            exec_datafusion_err!("st_buffer: invalid mitre_limit '{value}'")
                        })?;
                }
                "side" => {
                    if value != "both" {
                        return exec_err!("st_buffer: side={value} is not supported");
                    }
                }
                _ => return exec_err!("st_buffer: unknown style parameter '{key}'"),
            }
        }
        if let Join::Mitre { limit } = &mut style.join {
            *limit = mitre_limit;
        }
        Ok(style)
    }
}

// The shape added at the two ends of each linestring.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) enum EndCap {
    #[default]
    Round,
    // Cut straight across at the endpoint (PostGIS `flat`, alias `butt`).
    Flat,
    // A box extending `radius` past the endpoint.
    Square,
}

// The corner style at interior vertices of lines and polygon rings.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) enum Join {
    #[default]
    Round,
    // Sharp corner, beveled when the mitre tip would extend further than
    // `limit * radius` from the vertex.
    Mitre {
        limit: f64,
    },
    // Corner cut off with a single straight segment.
    Bevel,
}

// Per-segment arc angle for i_overlay's Round caps/joins, giving about
// `quad_segs` segments per quarter circle.
fn arc_angle(quad_segs: usize) -> f64 {
    std::f64::consts::FRAC_PI_2 / quad_segs.max(1) as f64
}

impl EndCap {
    fn to_line_cap(self, quad_segs: usize) -> LineCap<[f64; 2]> {
        match self {
            EndCap::Round => LineCap::Round(arc_angle(quad_segs)),
            EndCap::Flat => LineCap::Butt,
            EndCap::Square => LineCap::Square,
        }
    }
}

impl Join {
    fn to_line_join(self, quad_segs: usize) -> LineJoin<f64> {
        match self {
            Join::Round => LineJoin::Round(arc_angle(quad_segs)),
            // i_overlay takes the minimum interior angle instead of a length
            // ratio, unlike PostGIS. the limit is taken in PostGIS style
            // and converted to the i_overlay argument.
            Join::Mitre { limit } => LineJoin::Miter(2.0 * (1.0 / limit.max(1.0)).asin()),
            Join::Bevel => LineJoin::Bevel,
        }
    }
}

//*******************************************************
// Main logic to apply a buffer to a geometry.
//*******************************************************
// A geometry buffered in its crs units, written as WKB into `buf`.
pub(crate) fn geometry_buffer(
    bytes: &[u8],
    radius: f64,
    style: &BufferStyle,
    buf: &mut Vec<u8>,
) -> Result<()> {
    let geo = JoinableGeo::<Planar>::from_wkb(bytes).map_err(geo_err)?;
    let shapes = match &geo {
        JoinableGeo::Point { .. } => {
            let centers: Vec<[f64; 2]> = geo.dump_paths().into_iter().map(|p| p[0]).collect();
            buffer_points(&centers, radius, style)
        }
        JoinableGeo::Line { .. } => buffer_lines(geo.dump_paths(), radius, style),
        JoinableGeo::Poly { .. } => buffer_polys(geo.dump_paths(), radius, style),
    };
    write_shapes_wkb(shapes, buf)
}

// One polygon of i_overlay's output: a list of rings (outer first, then
// holes), each ring a list of vertices.
type Shape = Vec<Vec<[f64; 2]>>;

// A CCW ring approximating the circle of `radius` around `center`, with
// 4 * quad_segs vertices. The ring is open (the first vertex is not
// repeated at the end), which is what i_overlay expects as input.
fn circle([x, y]: [f64; 2], radius: f64, quad_segs: usize) -> Vec<[f64; 2]> {
    let n = 4 * quad_segs.max(1);
    let step = std::f64::consts::TAU / n as f64;
    (0..n)
        .map(|i| {
            let angle = i as f64 * step;
            [x + radius * angle.cos(), y + radius * angle.sin()]
        })
        .collect()
}

// A CCW axis-aligned square of half-side `radius` centered on `center`, open
// like `circle`'s rings.
fn square([x, y]: [f64; 2], radius: f64) -> Vec<[f64; 2]> {
    vec![
        [x - radius, y - radius],
        [x + radius, y - radius],
        [x + radius, y + radius],
        [x - radius, y + radius],
    ]
}

// The buffer of a point geo: one circle per point (a square with
// `endcap=square`), unioned when there are several (overlapping ones merge).
fn buffer_points(points: &[[f64; 2]], radius: f64, style: &BufferStyle) -> Vec<Shape> {
    if radius <= 0.0 {
        return Vec::new();
    }
    let mut rings: Vec<Vec<[f64; 2]>> = points
        .iter()
        .map(|&p| match style.endcap {
            EndCap::Square => square(p, radius),
            _ => circle(p, radius, style.quad_segs),
        })
        .collect();
    match rings.len() {
        0 => Vec::new(),
        // A lone circle is already a valid polygon; skip the union.
        1 => vec![vec![rings.pop().unwrap()]],
        // Single-input union: resolves the overlaps.
        _ => rings.simplify_shape(FillRule::NonZero),
    }
}

// The buffer of a line geo: i_overlay's stroke builds a band around each
// segment, the joins at interior vertices and the caps at each linestring's
// ends, then unions all the pieces.
fn buffer_lines(paths: Vec<Vec<[f64; 2]>>, radius: f64, style: &BufferStyle) -> Vec<Shape> {
    if radius <= 0.0 {
        return Vec::new();
    }
    let cap = style.endcap.to_line_cap(style.quad_segs);
    // The stroke width spans both sides of the line.
    let stroke_style = StrokeStyle::new(2.0 * radius)
        .line_join(style.join.to_line_join(style.quad_segs))
        .start_cap(cap.clone())
        .end_cap(cap);
    // Open paths: caps a linestring's ends even when it is closed.
    paths.stroke(stroke_style, false)
}

// The buffer of a poly geo from its rings (shells CCW, holes CW, as
// dump_paths gives them): i_overlay's outline offsets every ring with the
// requested joins and unions the pieces. A positive radius grows shells and
// shrinks holes, a negative one does the opposite. A zero radius only
// resolves overlaps and self-intersections.
fn buffer_polys(rings: Vec<Vec<[f64; 2]>>, radius: f64, style: &BufferStyle) -> Vec<Shape> {
    if radius == 0.0 {
        return rings.simplify_shape(FillRule::NonZero);
    }
    let outline_style =
        OutlineStyle::new(radius).line_join(style.join.to_line_join(style.quad_segs));
    rings.outline(&outline_style)
}

// One i_overlay shape (outer first, then holes) as a polygon. Polygon::new
// closes each ring (appends the first vertex if it isn't already repeated).
fn shape_to_polygon(shape: Shape) -> Polygon<f64> {
    let mut rings = shape.into_iter().map(LineString::from);
    let exterior = rings.next().unwrap_or_else(|| LineString::new(Vec::new()));
    Polygon::new(exterior, rings.collect())
}

// Writes i_overlay output as WKB into `buf` (cleared first): no shapes is an
// empty polygon, one shape a Polygon, several a MultiPolygon.
fn write_shapes_wkb(shapes: Vec<Shape>, buf: &mut Vec<u8>) -> Result<()> {
    buf.clear();
    let opts = WriteOptions::default();
    let written = match shapes.len() {
        0 => write_polygon(
            buf,
            &Polygon::new(LineString::new(Vec::new()), Vec::new()),
            &opts,
        ),
        1 => write_polygon(
            buf,
            &shape_to_polygon(shapes.into_iter().next().unwrap()),
            &opts,
        ),
        _ => write_multi_polygon(
            buf,
            &MultiPolygon::new(shapes.into_iter().map(shape_to_polygon).collect()),
            &opts,
        ),
    };
    written.map_err(|e| exec_datafusion_err!("st_buffer: {e}"))
}

//*******************************************************
// Main logic to apply a buffer to a geography.
//*******************************************************
// A geography buffered in meters: it's projected to an azimuthal equidistant
// projection centered on it, buffered there, and projected back. its edges are
// densified first, as they're great circles, which aren't straight in the
// projection. Then the geometry buffer logic is called.
pub(crate) fn geography_buffer(
    bytes: &[u8],
    radius: f64,
    style: &BufferStyle,
    buf: &mut Vec<u8>,
) -> Result<()> {
    let wkb = Wkb::try_new(bytes).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
    if wkb.dim() != Dimensions::Xy {
        return exec_err!("only 2D geometries are supported");
    }
    let Some(mut geom) = wkb.try_to_geometry() else {
        return geometry_buffer(bytes, radius, style, buf);
    };
    let Some(center) = vertex_center(&mut geom) else {
        return geometry_buffer(bytes, radius, style, buf);
    };

    // the same sphere for both, so that there's no datum shift between them.
    let proj_err = |e: ProjError| exec_datafusion_err!("{e}");
    let lonlat = Proj::from_proj_string(&format!("+proj=longlat +R={EARTH_RADIUS_METERS}"))
        .map_err(proj_err)?;
    let aeqd = Proj::from_proj_string(&format!(
        "+proj=aeqd +lat_0={} +lon_0={} +R={EARTH_RADIUS_METERS} +units=m",
        center.lat.deg(),
        center.lng.deg()
    ))
    .map_err(proj_err)?;
    let project = |x: f64, y: f64| transform_point(&lonlat, &aeqd, x, y, 0.0).map(|p| (p.0, p.1));

    // the error of the buffer's own arcs, which are chords of quad_segs per
    // quarter circle.
    let tol = radius.abs() * (1.0 - (PI / (4.0 * style.quad_segs as f64)).cos());
    if tol > 0.0 {
        densify_geometry(&mut geom, &project, tol).map_err(proj_err)?;
    }
    geom.transform_coordinates(&mut |x, y, z| transform_point(&lonlat, &aeqd, x, y, z))
        .map_err(proj_err)?;

    let opts = WriteOptions::default();
    let mut projected = Vec::new();
    write_geometry(&mut projected, &geom, &opts).map_err(|e| exec_datafusion_err!("{e}"))?;
    geometry_buffer(&projected, radius, style, buf)?;

    let buffered = Wkb::try_new(buf).map_err(|e| exec_datafusion_err!("malformed WKB: {e}"))?;
    let Some(mut buffered) = buffered.try_to_geometry() else {
        return Ok(());
    };
    buffered
        .transform_coordinates(&mut |x, y, z| transform_point(&aeqd, &lonlat, x, y, z))
        .map_err(proj_err)?;
    buf.clear();
    write_geometry(buf, &buffered, &opts).map_err(|e| exec_datafusion_err!("{e}"))
}

// The normalized mean of the vertices as unit vectors, or the first vertex if
// they nearly cancel out, e.g. a ring around the equator. None without
// vertices.
fn vertex_center(geom: &mut Geometry<f64>) -> Option<LatLng> {
    let mut sum = Vector {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut first = None;
    geom.transform_coordinates(&mut |x, y, z| {
        let p = Point::from(LatLng::from_degrees(y, x));
        first.get_or_insert(p);
        sum = sum + p.0;
        Ok((x, y, z))
    })
    .ok()?;
    let first = first?;
    if sum.norm() < 1e-6 {
        return Some(LatLng::from(first));
    }
    Some(LatLng::from(Point(sum.normalize())))
}

fn densify_geometry(
    geom: &mut Geometry<f64>,
    project: &dyn Fn(f64, f64) -> std::result::Result<(f64, f64), ProjError>,
    tol: f64,
) -> std::result::Result<(), ProjError> {
    match geom {
        Geometry::LineString(ls) => densify_line_string(ls, project, tol),
        Geometry::MultiLineString(mls) => mls
            .0
            .iter_mut()
            .try_for_each(|ls| densify_line_string(ls, project, tol)),
        Geometry::Polygon(poly) => densify_polygon(poly, project, tol),
        Geometry::MultiPolygon(mp) => {
            mp.0.iter_mut()
                .try_for_each(|poly| densify_polygon(poly, project, tol))
        }
        Geometry::GeometryCollection(gc) => {
            gc.0.iter_mut()
                .try_for_each(|g| densify_geometry(g, project, tol))
        }
        _ => Ok(()),
    }
}

fn densify_polygon(
    poly: &mut Polygon<f64>,
    project: &dyn Fn(f64, f64) -> std::result::Result<(f64, f64), ProjError>,
    tol: f64,
) -> std::result::Result<(), ProjError> {
    let empty = Polygon::new(LineString::new(Vec::new()), Vec::new());
    let (mut exterior, mut interiors) = std::mem::replace(poly, empty).into_inner();
    densify_line_string(&mut exterior, project, tol)?;
    for ring in &mut interiors {
        densify_line_string(ring, project, tol)?;
    }
    *poly = Polygon::new(exterior, interiors);
    Ok(())
}

fn densify_line_string(
    ls: &mut LineString<f64>,
    project: &dyn Fn(f64, f64) -> std::result::Result<(f64, f64), ProjError>,
    tol: f64,
) -> std::result::Result<(), ProjError> {
    let mut out = Vec::with_capacity(ls.0.len());
    let mut prev: Option<(Coord, (f64, f64))> = None;
    for &c in &ls.0 {
        let pc = project(c.x, c.y)?;
        if let Some((a, pa)) = prev {
            densify_spherical_edge(a, pa, c, pc, project, tol, 0, &mut out)?;
        }
        out.push(c);
        prev = Some((c, pc));
    }
    ls.0 = out;
    Ok(())
}

// Bounds the recursion, e.g. for an edge near the projection's antipode, where
// it's badly distorted.
const MAX_DENSIFY_DEPTH: u32 = 16;

// Appends the points strictly between `a` and `b`, so that each projected
// sub-edge is within `tol` of its projected great circle. `pa` and `pb` are
// the projected `a` and `b`.
#[allow(clippy::too_many_arguments)]
fn densify_spherical_edge(
    a: Coord,
    pa: (f64, f64),
    b: Coord,
    pb: (f64, f64),
    project: &dyn Fn(f64, f64) -> std::result::Result<(f64, f64), ProjError>,
    tol: f64,
    depth: u32,
    out: &mut Vec<Coord>,
) -> std::result::Result<(), ProjError> {
    if depth == MAX_DENSIFY_DEPTH {
        return Ok(());
    }
    let to_point = |c: Coord| Point::from(LatLng::from_degrees(c.y, c.x));
    let mid = LatLng::from(interpolate(0.5, &to_point(a), &to_point(b)));
    let mid = Coord {
        x: mid.lng.deg(),
        y: mid.lat.deg(),
    };
    let pm = project(mid.x, mid.y)?;

    // the distance from the projected midpoint to the projected chord.
    let (dx, dy) = (pb.0 - pa.0, pb.1 - pa.1);
    let len = dx.hypot(dy);
    let deviation = if len == 0.0 {
        (pm.0 - pa.0).hypot(pm.1 - pa.1)
    } else {
        (dx * (pm.1 - pa.1) - dy * (pm.0 - pa.0)).abs() / len
    };
    if deviation <= tol {
        return Ok(());
    }

    densify_spherical_edge(a, pa, mid, pm, project, tol, depth + 1, out)?;
    out.push(mid);
    densify_spherical_edge(mid, pm, b, pb, project, tol, depth + 1, out)
}
