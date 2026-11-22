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

use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use itertools::Itertools;
use s2::latlng::LatLng;
use s2::point::Point as S2Point;
use s2::r3::vector::Vector;
use s2::rect_bounder::RectBounder;
use wkb::reader::Wkb;

use super::raw_geo::EARTH_RADIUS_METERS;

pub(crate) const EPS: f64 = 1e-10;

// Latitude of the north pole (negated for the south pole), in degrees.
const POLE_LAT: f64 = 90.0;

// Cheap geometry-validity errors caught at construction (not full
// DE-9IM validation, just the checks that prevent later misbehavior
// like divide-by-zero).
#[derive(Debug)]
pub(crate) enum GeoError {
    // The WKB bytes could not be parsed.
    MalformedWkb(String),
    // A segment/edge with identical endpoints (zero length).
    DegenerateSegment,
    // A spherical segment/edge with (near) antipodal endpoints, for which the
    // great circle is undefined.
    AntipodalSegment,
    // A polygon ring whose first and last vertices differ (or that
    // produced no edges).
    UnclosedRing,
    // A spatial join was given JoinSide::None, which has no valid operand ordering.
    InvalidJoinSide,
}

impl std::fmt::Display for GeoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoError::MalformedWkb(msg) => write!(f, "malformed WKB: {msg}"),
            GeoError::DegenerateSegment => write!(f, "degenerate segment (identical endpoints)"),
            GeoError::AntipodalSegment => {
                write!(f, "antipodal segment (great circle is undefined)")
            }
            GeoError::UnclosedRing => write!(f, "polygon ring is not closed"),
            GeoError::InvalidJoinSide => {
                write!(f, "JoinSide::None is not valid for a spatial join")
            }
        }
    }
}

impl std::error::Error for GeoError {}

//*************************************************
// A simple bbox that well be inserted in a natural spatial index.
//*************************************************
#[derive(Copy, Clone, Debug)]
pub(crate) struct IndexBox {
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
}

// Implemented by both internal index nodes and leaf components.
pub(crate) trait Bboxable {
    fn bbox(&self) -> IndexBox;

    // Symmetric bbox overlap (EPS-slackened).
    fn box_overlap(&self, other: &impl Bboxable) -> bool {
        let a = self.bbox();
        let b = other.bbox();
        a.min_x <= b.max_x + EPS
            && a.max_x >= b.min_x - EPS
            && a.min_y <= b.max_y + EPS
            && a.max_y >= b.min_y - EPS
    }
}

impl Bboxable for IndexBox {
    fn bbox(&self) -> IndexBox {
        *self
    }
}

//*************************************************
// The base components that all geometries are built from, as
// traits so that the geometry logic is shared between the planar
// and spherical cases.
//*************************************************
pub(crate) trait Point: Bboxable + Copy {
    fn new(x: f64, y: f64) -> Self;
    fn is_equal(&self, other: &Self) -> bool;

    // The coordinates the point was created with.
    fn xy(&self) -> [f64; 2];
}

// What line segments and polygon edges have in common.
pub(crate) trait Segment: Bboxable + Copy {
    type P: Point;

    fn p1(&self) -> &Self::P;
    fn p2(&self) -> &Self::P;
}

pub(crate) trait LineSegment: Segment {
    fn new(p1: Self::P, p2: Self::P, p1_boundary: bool, p2_boundary: bool) -> Self;
    fn p1_boundary(&self) -> bool;
    fn p2_boundary(&self) -> bool;
    fn set_boundaries(&mut self, p1_boundary: bool, p2_boundary: bool);
}

pub(crate) trait Edge: Segment {
    fn new(p1: Self::P, p2: Self::P, next: usize, interior_on_left: bool) -> Self;

    // Index (in the geometry's edge vector) of the next edge in this ring;
    // wraps to the ring's first edge at the ring end.
    fn next(&self) -> usize;
    fn interior_on_left(&self) -> bool;
    fn set_next(&mut self, next: usize);
    fn set_interior_on_left(&mut self, interior_on_left: bool);
}

//*************************************************
// The geometric primitives the helpers are built on, implemented
// by each kernel.
//*************************************************
pub(crate) trait GeoKernel {
    type Point: Point;
    type Line: LineSegment<P = Self::Point>;
    type Edge: Edge<P = Self::Point>;
    type Dir: Copy;

    // Distances are compared as an internal squared measure, cheap to compute and
    // increasing with the distance. These convert a distance, in the unit the
    // caller works in, to and from it.
    fn to_internal_dist_sq(dist: f64) -> f64;
    fn from_internal_dist_sq(dist_sq: f64) -> f64;

    fn dist_sq(p: &Self::Point, q: &Self::Point) -> f64;

    // Squared distance from p to the closest point of the segment.
    fn dist_sq_to_segment<S: Segment<P = Self::Point>>(p: &Self::Point, seg: &S) -> f64;

    // Squared minimum distance between two segments that don't intersect, which
    // is reached at one of their endpoints.
    fn dist_sq_to_disjoint<A: Segment<P = Self::Point>, B: Segment<P = Self::Point>>(
        a: &A,
        b: &B,
    ) -> f64 {
        Self::dist_sq_to_segment(a.p1(), b)
            .min(Self::dist_sq_to_segment(a.p2(), b))
            .min(Self::dist_sq_to_segment(b.p1(), a))
            .min(Self::dist_sq_to_segment(b.p2(), a))
    }

    // Whether a and b can form a segment.
    fn check_segment(a: &Self::Point, b: &Self::Point) -> Result<(), GeoError>;

    // Whether p is on the line through the segment.
    fn is_collinear<S: Segment<P = Self::Point>>(seg: &S, p: &Self::Point) -> bool;

    // The parameter of p's projection on the segment, 0 at p1 and 1 at p2.
    fn projection_t<S: Segment<P = Self::Point>>(seg: &S, p: &Self::Point) -> f64;

    fn midpoint<S: Segment<P = Self::Point>>(seg: &S) -> Self::Point;

    // Where the lines through the two non-parallel segments meet, as parameters
    // on each segment. None if they're parallel. the parameters aren't
    // restricted to the segments, it's up to the caller.
    fn intersection<A: Segment<P = Self::Point>, B: Segment<P = Self::Point>>(
        a: &A,
        b: &B,
    ) -> Option<SegmentIntersection>;

    // Whether the two segments point the same way.
    fn same_direction<A: Segment<P = Self::Point>, B: Segment<P = Self::Point>>(
        a: &A,
        b: &B,
    ) -> bool;

    // The direction of travel at `at` along the segment from `from` to `to`
    // (`at` being on that segment's line).
    fn dir_at(from: &Self::Point, to: &Self::Point, at: &Self::Point) -> Self::Dir;

    // The signed angle at `at` from d1 to d2, positive counterclockwise.
    fn turn(at: &Self::Point, d1: &Self::Dir, d2: &Self::Dir) -> f64;

    // The angle at edge's end vertex, measured inside the polygon, from the
    // edge towards `dir`.
    fn angle_to(edge: &Self::Edge, dir: &Self::Dir) -> f64 {
        let v = edge.p2();
        let raw_turn = Self::turn(v, &Self::dir_at(edge.p1(), v, v), dir);
        if edge.interior_on_left() {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        }
    }

    // Interior angle of the polygon at edge's end vertex.
    fn vertex_angle(edge: &Self::Edge, next: &Self::Edge) -> f64 {
        Self::angle_to(edge, &Self::dir_at(next.p1(), next.p2(), edge.p2()))
    }

    // Whether edge's end vertex is reflex (interior angle > pi), the interior
    // wedge being formed with next.
    fn is_reflex(edge: &Self::Edge, next: &Self::Edge) -> bool {
        Self::vertex_angle(edge, next) > std::f64::consts::PI
    }

    // For a line through edge's end vertex, whether the forward ray and
    // the opposite ray each lie strictly inside the interior wedge formed
    // by edge and next.
    fn wedge_membership(
        edge: &Self::Edge,
        next: &Self::Edge,
        from: &Self::Point,
        to: &Self::Point,
    ) -> (bool, bool) {
        let angle = Self::vertex_angle(edge, next);
        let seg_angle = Self::angle_to(edge, &Self::dir_at(from, to, edge.p2()));
        let flipped = if seg_angle < std::f64::consts::PI {
            seg_angle + std::f64::consts::PI
        } else {
            seg_angle - std::f64::consts::PI
        };
        let seg_in_wedge = seg_angle > EPS && seg_angle < angle - EPS;
        let flipped_in_wedge = flipped > EPS && flipped < angle - EPS;
        (seg_in_wedge, flipped_in_wedge)
    }

    // Whether the polygon interior is on the left of the closed ring's (first
    // vertex repeated last) edges.
    fn interior_on_left(ring: &[Self::Point], is_exterior: bool) -> bool;

    // Whether the ray cast from p crosses edge. an edge owns its end vertex,
    // not its start.
    fn ray_crosses_edge(p: &Self::Point, edge: &Self::Edge, next: &Self::Edge) -> bool;

    // The distance along the ray cast from p to where it crosses edge. None if
    // it doesn't.
    fn ray_crossing_distance(p: &Self::Point, edge: &Self::Edge) -> Option<f64>;

    // Index pruning for ray casting: whether a ray cast from anywhere in a could
    // reach b. slackened by EPS so it's never tighter than the exact tests.
    fn ray_candidate(a: &IndexBox, b: &IndexBox) -> bool;

    // A lower bound of the squared distance between anything in a and anything
    // in b (0 when they overlap), in the same unit as dist_sq.
    fn box_min_dist_sq(a: &IndexBox, b: &IndexBox) -> f64;

    // Whether a poly's shell can bound a part of its complement that another poly
    // could contain. Only holes can on the plane, the outside of a shell being
    // unbounded, while on the sphere the outside of a shell is bounded too.
    const SHELLS_BOUND_COMPLEMENT: bool;

    // Whether the reference point, toward which all rays are cast, is inside the
    // geometry, and in which poly. a point is inside when this differs from
    // whether the ray from it to the reference point crosses the edges an odd
    // number of times. e.g. reference outside (false) and odd crossings (true) ->
    // inside. edges[holes_start..] are the hole rings' edges.
    fn reference_status(edges: &[Self::Edge], poly_ids: &[usize], holes_start: usize) -> RefStatus;

    // For a ring going through the reference point, whether it encloses the point
    // just short of the reference along the ray cast from p.
    fn ring_encloses_ray_end(edges: &[Self::Edge], ring: &Ring, p: &Self::Point) -> bool;

    // The bbox of everything a poly geometry covers, from the union of its edges'
    // bboxes.
    fn poly_bbox(edges: &[Self::Edge], edges_box: IndexBox, ref_status: &RefStatus) -> IndexBox;
}

// Which poly of a poly geometry the reference point is in, if any. For valid
// geometries (holes inside their shell, polys with disjoint interiors), a point is
// in a poly iff an odd number of its rings enclose it, so this is an XOR over each
// poly's rings, at most one poly having it true.
//
// A ring going through the reference doesn't give it a status. Rays then stop just
// short of the reference, a point whose status depends on the ray's direction, so
// those rings are kept to be checked per ray. A poly whose shell goes through it
// has no other ring enclosing it, and a poly with only holes going through it has
// a shell enclosing it, so for each poly the rings that don't go through it amount
// to whether it's ref_poly.
pub(crate) struct RefStatus {
    // The poly whose rings that don't go through the reference enclose it.
    ref_poly: Option<usize>,
    pole_rings: Vec<Ring>,

    // Spherical only. Whether the south pole's status differs from the north
    // pole's, i.e. an odd number of rings separate them, over the rings that don't
    // go through the north pole.
    poles_differ: bool,
}

impl RefStatus {
    // Whether the end of every ray is outside, as for every planar geometry.
    pub(crate) fn always_outside(&self) -> bool {
        self.ref_poly.is_none() && self.pole_rings.is_empty()
    }

    // The poly the end of the ray cast from p is in, if any.
    pub(crate) fn containing_poly_for<K: GeoKernel>(
        &self,
        edges: &[K::Edge],
        p: &K::Point,
    ) -> Option<usize> {
        if self.pole_rings.is_empty() {
            return self.ref_poly;
        }
        let contains = |poly: usize| {
            self.pole_rings
                .iter()
                .filter(|ring| ring.poly_id == poly)
                .fold(self.ref_poly == Some(poly), |inside, ring| {
                    inside ^ K::ring_encloses_ray_end(edges, ring, p)
                })
        };
        self.ref_poly
            .into_iter()
            .chain(self.pole_rings.iter().map(|ring| ring.poly_id))
            .find(|&poly| contains(poly))
    }

    // Whether the end of the ray cast from p is inside.
    pub(crate) fn inside_for<K: GeoKernel>(&self, edges: &[K::Edge], p: &K::Point) -> bool {
        self.containing_poly_for::<K>(edges, p).is_some()
    }
}

pub(crate) struct SegmentIntersection {
    pub(crate) t: f64,
    pub(crate) u: f64,
    // Whether the second segment's direction is clockwise from the first's.
    pub(crate) clockwise: bool,
}

//*************************************************
// Planar components.
//*************************************************
#[derive(Copy, Clone, Debug)]
pub(crate) struct PlanarPoint {
    x: f64,
    y: f64,
}

impl PlanarPoint {
    fn diff(&self, other: &PlanarPoint) -> Dir {
        Dir {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Point for PlanarPoint {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn is_equal(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }

    fn xy(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

impl Bboxable for PlanarPoint {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.x,
            min_y: self.y,
            max_x: self.x,
            max_y: self.y,
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct Dir {
    x: f64,
    y: f64,
}

impl Dir {
    fn cross(&self, other: &Dir) -> f64 {
        self.x * other.y - self.y * other.x
    }

    fn dot(&self, other: &Dir) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn turn_to(&self, other: &Dir) -> f64 {
        self.cross(other).atan2(self.dot(other))
    }
}

fn dir<S: Segment<P = PlanarPoint>>(seg: &S) -> Dir {
    seg.p2().diff(seg.p1())
}

fn planar_bbox(p1: &PlanarPoint, p2: &PlanarPoint) -> IndexBox {
    IndexBox {
        min_x: p1.x.min(p2.x),
        min_y: p1.y.min(p2.y),
        max_x: p1.x.max(p2.x),
        max_y: p1.y.max(p2.y),
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PlanarLineSegment {
    p1: PlanarPoint,
    p2: PlanarPoint,
    p1_boundary: bool,
    p2_boundary: bool,
}

impl Segment for PlanarLineSegment {
    type P = PlanarPoint;

    fn p1(&self) -> &PlanarPoint {
        &self.p1
    }

    fn p2(&self) -> &PlanarPoint {
        &self.p2
    }
}

impl Bboxable for PlanarLineSegment {
    fn bbox(&self) -> IndexBox {
        planar_bbox(&self.p1, &self.p2)
    }
}

impl LineSegment for PlanarLineSegment {
    fn new(p1: PlanarPoint, p2: PlanarPoint, p1_boundary: bool, p2_boundary: bool) -> Self {
        Self {
            p1,
            p2,
            p1_boundary,
            p2_boundary,
        }
    }

    fn p1_boundary(&self) -> bool {
        self.p1_boundary
    }

    fn p2_boundary(&self) -> bool {
        self.p2_boundary
    }

    fn set_boundaries(&mut self, p1_boundary: bool, p2_boundary: bool) {
        self.p1_boundary = p1_boundary;
        self.p2_boundary = p2_boundary;
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PlanarEdge {
    p1: PlanarPoint,
    p2: PlanarPoint,
    next: usize,
    interior_on_left: bool,
}

impl Segment for PlanarEdge {
    type P = PlanarPoint;

    fn p1(&self) -> &PlanarPoint {
        &self.p1
    }

    fn p2(&self) -> &PlanarPoint {
        &self.p2
    }
}

impl Bboxable for PlanarEdge {
    fn bbox(&self) -> IndexBox {
        planar_bbox(&self.p1, &self.p2)
    }
}

impl Edge for PlanarEdge {
    fn new(p1: PlanarPoint, p2: PlanarPoint, next: usize, interior_on_left: bool) -> Self {
        Self {
            p1,
            p2,
            next,
            interior_on_left,
        }
    }

    fn next(&self) -> usize {
        self.next
    }

    fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    fn set_next(&mut self, next: usize) {
        self.next = next;
    }

    fn set_interior_on_left(&mut self, interior_on_left: bool) {
        self.interior_on_left = interior_on_left;
    }
}

// The x-coordinate where the segment reaches height p.y, or None when p.y is
// outside the segment's y-span or the segment is horizontal.
fn x_intercept(p1: &PlanarPoint, p2: &PlanarPoint, p: &PlanarPoint) -> Option<f64> {
    let dy = p2.y - p1.y;
    if dy == 0.0 {
        return None;
    }
    let t = (p.y - p1.y) / dy;
    (0.0..=1.0).contains(&t).then_some(p1.x + t * (p2.x - p1.x))
}

//*************************************************
// Planar implementation of the GeoKernal.
//*************************************************
pub(crate) struct Planar;

impl GeoKernel for Planar {
    type Point = PlanarPoint;
    type Line = PlanarLineSegment;
    type Edge = PlanarEdge;
    type Dir = Dir;

    const SHELLS_BOUND_COMPLEMENT: bool = false;

    fn to_internal_dist_sq(dist: f64) -> f64 {
        dist * dist
    }

    fn from_internal_dist_sq(dist_sq: f64) -> f64 {
        dist_sq.sqrt()
    }

    fn dist_sq(p: &PlanarPoint, q: &PlanarPoint) -> f64 {
        let d = p.diff(q);
        d.dot(&d)
    }

    fn dist_sq_to_segment<S: Segment<P = PlanarPoint>>(p: &PlanarPoint, seg: &S) -> f64 {
        let t = Self::projection_t(seg, p).clamp(0.0, 1.0);
        let (p1, d) = (seg.p1(), dir(seg));
        let closest = PlanarPoint {
            x: p1.x + t * d.x,
            y: p1.y + t * d.y,
        };
        Self::dist_sq(p, &closest)
    }

    fn check_segment(a: &PlanarPoint, b: &PlanarPoint) -> Result<(), GeoError> {
        if a.is_equal(b) {
            return Err(GeoError::DegenerateSegment);
        }
        Ok(())
    }

    fn is_collinear<S: Segment<P = PlanarPoint>>(seg: &S, p: &PlanarPoint) -> bool {
        let d = dir(seg);
        let c = d.cross(&p.diff(seg.p1()));
        c * c < EPS * EPS * d.dot(&d)
    }

    fn projection_t<S: Segment<P = PlanarPoint>>(seg: &S, p: &PlanarPoint) -> f64 {
        let d = dir(seg);
        d.dot(&p.diff(seg.p1())) / d.dot(&d)
    }

    fn midpoint<S: Segment<P = PlanarPoint>>(seg: &S) -> PlanarPoint {
        let (a, b) = (seg.p1(), seg.p2());
        PlanarPoint {
            x: (a.x + b.x) / 2.0,
            y: (a.y + b.y) / 2.0,
        }
    }

    fn intersection<A: Segment<P = PlanarPoint>, B: Segment<P = PlanarPoint>>(
        a: &A,
        b: &B,
    ) -> Option<SegmentIntersection> {
        let da = dir(a);
        let db = dir(b);
        let cross = da.cross(&db);
        if cross.abs() < EPS {
            return None;
        }
        let f = b.p1().diff(a.p1());
        Some(SegmentIntersection {
            t: -db.cross(&f) / cross,
            u: -da.cross(&f) / cross,
            clockwise: cross < 0.0,
        })
    }

    fn same_direction<A: Segment<P = PlanarPoint>, B: Segment<P = PlanarPoint>>(
        a: &A,
        b: &B,
    ) -> bool {
        dir(a).dot(&dir(b)) > 0.0
    }

    fn dir_at(from: &PlanarPoint, to: &PlanarPoint, _at: &PlanarPoint) -> Dir {
        to.diff(from)
    }

    fn turn(_at: &PlanarPoint, d1: &Dir, d2: &Dir) -> f64 {
        d1.turn_to(d2)
    }

    // Shells are interior on the left when ccw, holes when cw.
    fn interior_on_left(ring: &[PlanarPoint], is_exterior: bool) -> bool {
        let signed_area: f64 = ring
            .windows(2)
            .map(|w| w[0].x * w[1].y - w[1].x * w[0].y)
            .sum();
        (signed_area > 0.0) == is_exterior
    }

    fn ray_crosses_edge(p: &PlanarPoint, edge: &PlanarEdge, next: &PlanarEdge) -> bool {
        let (p1, p2) = (edge.p1(), edge.p2());
        // Check the owned end vertex first: for a horizontal edge sitting on the ray
        // (p1.y == p2.y == p.y) this resolves its p2 vertex via the wedge test.
        if p.y == p2.y {
            if p2.x <= p.x {
                return false;
            }
            return crossing_at_vertex::<Planar>(edge, next, p, p2).is_some();
        }
        if p.y == p1.y {
            return false; // first endpoint: owned by the previous edge
        }
        x_intercept(p1, p2, p).is_some_and(|xi| xi > p.x)
    }

    fn ray_crossing_distance(p: &PlanarPoint, edge: &PlanarEdge) -> Option<f64> {
        x_intercept(edge.p1(), edge.p2(), p).map(|xi| xi - p.x)
    }

    // The ray goes in +x, so b must share a's y-band and not be entirely left of a.
    fn ray_candidate(a: &IndexBox, b: &IndexBox) -> bool {
        a.min_y <= b.max_y + EPS && a.max_y >= b.min_y - EPS && b.max_x >= a.min_x - EPS
    }

    fn box_min_dist_sq(a: &IndexBox, b: &IndexBox) -> f64 {
        let dx = (a.min_x - b.max_x).max(b.min_x - a.max_x).max(0.0);
        let dy = (a.min_y - b.max_y).max(b.min_y - a.max_y).max(0.0);
        dx * dx + dy * dy
    }

    // Rays go to infinity, which is outside every polygon.
    fn reference_status(
        _edges: &[PlanarEdge],
        _poly_ids: &[usize],
        _holes_start: usize,
    ) -> RefStatus {
        RefStatus {
            ref_poly: None,
            pole_rings: Vec::new(),
            poles_differ: false,
        }
    }

    fn ring_encloses_ray_end(_edges: &[PlanarEdge], _ring: &Ring, _p: &PlanarPoint) -> bool {
        unreachable!("the planar reference is never on a boundary")
    }

    fn poly_bbox(_edges: &[PlanarEdge], edges_box: IndexBox, _ref_status: &RefStatus) -> IndexBox {
        edges_box
    }
}

//*************************************************
// Spherical components, for geographies. segments and edges store their
// bbox since computing it accounts for the geodesic edge, which is too
// expensive to compute on the fly.
//*************************************************
#[derive(Copy, Clone, Debug)]
pub(crate) struct SphericalPoint {
    x: f64,
    y: f64,
    xyz: S2Point,
}

// The poles and the antimeridian have several lon/lat representations. they're
// normalized (lon 0 at the poles, 180 rather than -180) so that is_equal can
// compare coordinates exactly.
fn normalize_lon_lat(x: f64, y: f64) -> (f64, f64) {
    if y.abs() == POLE_LAT {
        (0.0, y)
    } else if x == -180.0 {
        (180.0, y)
    } else {
        (x, y)
    }
}

impl SphericalPoint {
    fn from_xyz(xyz: S2Point) -> Self {
        let ll = LatLng::from(&xyz);
        let (x, y) = normalize_lon_lat(ll.lng.deg(), ll.lat.deg());
        Self { x, y, xyz }
    }
}

// The unit normal of the great circle through the segment.
fn unit_normal<S: Segment<P = SphericalPoint>>(seg: &S) -> Vector {
    seg.p1().xyz.cross(&seg.p2().xyz).0.normalize()
}

impl Point for SphericalPoint {
    fn new(x: f64, y: f64) -> Self {
        let (x, y) = normalize_lon_lat(x, y);
        Self {
            x,
            y,
            xyz: S2Point::from(LatLng::from_degrees(y, x)),
        }
    }

    fn is_equal(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }

    fn xy(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

impl Bboxable for SphericalPoint {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.x,
            min_y: self.y,
            max_x: self.x,
            max_y: self.y,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct SphericalLineSegment {
    p1: SphericalPoint,
    p2: SphericalPoint,
    p1_boundary: bool,
    p2_boundary: bool,
    bbox: IndexBox,
}

impl Segment for SphericalLineSegment {
    type P = SphericalPoint;

    fn p1(&self) -> &SphericalPoint {
        &self.p1
    }

    fn p2(&self) -> &SphericalPoint {
        &self.p2
    }
}

impl Bboxable for SphericalLineSegment {
    fn bbox(&self) -> IndexBox {
        self.bbox
    }
}

impl LineSegment for SphericalLineSegment {
    fn new(p1: SphericalPoint, p2: SphericalPoint, p1_boundary: bool, p2_boundary: bool) -> Self {
        Self {
            p1,
            p2,
            p1_boundary,
            p2_boundary,
            bbox: geodesic_bbox(&p1, &p2),
        }
    }

    fn p1_boundary(&self) -> bool {
        self.p1_boundary
    }

    fn p2_boundary(&self) -> bool {
        self.p2_boundary
    }

    fn set_boundaries(&mut self, p1_boundary: bool, p2_boundary: bool) {
        self.p1_boundary = p1_boundary;
        self.p2_boundary = p2_boundary;
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct SphericalEdge {
    p1: SphericalPoint,
    p2: SphericalPoint,
    next: usize,
    interior_on_left: bool,
    bbox: IndexBox,
}

impl Segment for SphericalEdge {
    type P = SphericalPoint;

    fn p1(&self) -> &SphericalPoint {
        &self.p1
    }

    fn p2(&self) -> &SphericalPoint {
        &self.p2
    }
}

impl Bboxable for SphericalEdge {
    fn bbox(&self) -> IndexBox {
        self.bbox
    }
}

impl Edge for SphericalEdge {
    fn new(p1: SphericalPoint, p2: SphericalPoint, next: usize, interior_on_left: bool) -> Self {
        Self {
            p1,
            p2,
            next,
            interior_on_left,
            bbox: geodesic_bbox(&p1, &p2),
        }
    }

    fn next(&self) -> usize {
        self.next
    }

    fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    fn set_next(&mut self, next: usize) {
        self.next = next;
    }

    fn set_interior_on_left(&mut self, interior_on_left: bool) {
        self.interior_on_left = interior_on_left;
    }
}

// Wraps a longitude difference into (-180, 180].
pub(crate) fn wrap_lon_diff(d: f64) -> f64 {
    if d > 180.0 {
        d - 360.0
    } else if d <= -180.0 {
        d + 360.0
    } else {
        d
    }
}

// The latitude where the edge reaches longitude p.x, or None when p.x is
// outside the edge's longitude span or the edge lies along a meridian (same
// longitude at both ends, a pole endpoint, or passing over a pole). An edge
// shorter than 180 degrees that doesn't pass over a pole spans less than 180
// degrees of longitude, so its span is the short way around.
fn lat_intercept(p1: &SphericalPoint, p2: &SphericalPoint, p: &SphericalPoint) -> Option<f64> {
    let is_pole = |q: &SphericalPoint| q.y.abs() == POLE_LAT;
    let d_edge = wrap_lon_diff(p2.x - p1.x);
    if d_edge == 0.0 || d_edge == 180.0 || is_pole(p1) || is_pole(p2) {
        return None;
    }
    let t = wrap_lon_diff(p.x - p1.x) / d_edge;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    let n = p1.xyz.cross(&p2.xyz).0;
    let lon = p.x.to_radians();
    Some(
        (-(n.x * lon.cos() + n.y * lon.sin()) / n.z)
            .atan()
            .to_degrees(),
    )
}

// The bbox of the geodesic edge between two points, in lon/lat degrees. an edge
// crossing the antimeridian gets the full longitude range.
fn geodesic_bbox(p1: &SphericalPoint, p2: &SphericalPoint) -> IndexBox {
    let mut bounder = RectBounder::new();
    bounder.add_point(&p1.xyz);
    bounder.add_point(&p2.xyz);
    let bound = bounder.get_bound();
    let (min_x, max_x) = if bound.is_inverted() {
        (-180.0, 180.0)
    } else {
        (bound.lng_lo().deg(), bound.lng_hi().deg())
    };
    IndexBox {
        min_x,
        min_y: bound.lat_lo().deg(),
        max_x,
        max_y: bound.lat_hi().deg(),
    }
}

//*************************************************
// A helper struct that handles rings and all the pole and
// reference point logic for ray casting.
//
// A ring is edges[start..=end] of a poly geometry, the last edge's next wrapping
// to the first. Since interiors are on the left of every edge, the area a ring
// encloses is the one on its left for a shell, and the one on its right for a
// hole.
//*************************************************
#[derive(Copy, Clone, Debug)]
pub(crate) struct Ring {
    start: usize,
    end: usize,
    poly_id: usize,
    is_hole: bool,

    // The edge that ends at the north pole, or goes over it, if the ring goes
    // through it.
    pole_edge: Option<usize>,

    // Whether the ring goes around the globe, putting the poles on opposite sides
    // of it. Meaningless for a ring going through a pole.
    separates_poles: bool,
}

impl Ring {
    // The rings of a poly geometry, edges[holes_start..] being the holes' edges.
    fn all(edges: &[SphericalEdge], poly_ids: &[usize], holes_start: usize) -> Vec<Ring> {
        let mut rings = Vec::new();
        let mut start = 0;
        while start < edges.len() {
            let mut end = start;
            while edges[end].next() == end + 1 {
                end += 1;
            }
            rings.push(Ring {
                start,
                end,
                poly_id: poly_ids[start],
                is_hole: start >= holes_start,
                pole_edge: (start..=end).find(|&i| Self::goes_through_north_pole(&edges[i])),
                separates_poles: Self::separates_poles(&edges[start..=end]),
            });
            start = end + 1;
        }
        rings
    }

    // An edge starting at the pole isn't counted, the previous one is. An edge
    // goes over a pole when its endpoints are 180 degrees of longitude apart, the
    // nearer pole.
    fn goes_through_north_pole(edge: &SphericalEdge) -> bool {
        let (p1, p2) = (edge.p1(), edge.p2());
        if p2.y == POLE_LAT {
            return true;
        }
        p1.y.abs() != POLE_LAT
            && p2.y.abs() != POLE_LAT
            && wrap_lon_diff(p2.x - p1.x) == 180.0
            && p1.y + p2.y > 0.0
    }

    // The ring's longitude changes add up to +-360 when it goes around the globe,
    // 0 otherwise. Each edge's change is the short way around, which is the way it
    // goes, short of going over a pole.
    fn separates_poles(ring_edges: &[SphericalEdge]) -> bool {
        let total: f64 = ring_edges
            .iter()
            .map(|edge| wrap_lon_diff(edge.p2().x - edge.p1().x))
            .sum();
        total.abs() > 180.0
    }

    // For a ring that doesn't go through the north pole. The ray is cast from the
    // ring's northmost vertex v, from just north of it actually, which is on the
    // ring's left iff north is in v's interior wedge. That's never ambiguous, an
    // edge going north from v would reach a vertex further north. v's edges are
    // left out of the ray casting, the wedge accounts for them.
    fn encloses_north_pole(&self, edges: &[SphericalEdge]) -> bool {
        let vi = (self.start..=self.end)
            .max_by(|&a, &b| edges[a].p2().y.total_cmp(&edges[b].p2().y))
            .expect("a ring has edges");
        let edge = &edges[vi];
        let next_i = edge.next();
        let v = edge.p2();
        let north = Spherical::dir_at(v, &SphericalPoint::new(0.0, POLE_LAT), v);
        let start_on_left =
            Spherical::angle_to(edge, &north) < Spherical::vertex_angle(edge, &edges[next_i]);
        let crossings = (self.start..=self.end)
            .filter(|&i| i != vi && i != next_i)
            .filter(|&i| Spherical::ray_crosses_edge(v, &edges[i], &edges[edges[i].next()]))
            .count();
        let pole_on_left = start_on_left != (crossings % 2 == 1);
        pole_on_left != self.is_hole
    }

    // For a ring that goes through the north pole, whether it encloses the point
    // just short of the pole along the ray cast from p, i.e. on p's meridian. Its
    // direction from the pole is that meridian's, which is in the interior wedge
    // at a pole vertex, or on the left of an edge going over the pole.
    fn encloses_ray_end(&self, edges: &[SphericalEdge], p: &SphericalPoint) -> bool {
        let edge = &edges[self.pole_edge.expect("a ring going through the pole")];
        let lon = p.x.to_radians();
        let dir = Vector::new(lon.cos(), lon.sin(), 0.0);
        let on_left = if edge.p2().y == POLE_LAT {
            let angle = Spherical::angle_to(edge, &dir);
            angle > EPS && angle < Spherical::vertex_angle(edge, &edges[edge.next()]) - EPS
        } else {
            let n = edge.p1().xyz.cross(&edge.p2().xyz).0;
            n.dot(&dir) > 0.0
        };
        on_left != self.is_hole
    }
}

//*************************************************
// Spherical implementation of the GeoKernal.
//*************************************************
pub(crate) struct Spherical;

impl GeoKernel for Spherical {
    type Point = SphericalPoint;
    type Line = SphericalLineSegment;
    type Edge = SphericalEdge;
    type Dir = Vector;

    const SHELLS_BOUND_COMPLEMENT: bool = true;

    // The internal measure is the squared chord length between points on the unit
    // sphere, for an angle θ between them: (2 sin(θ / 2))². Distances are in meters.
    fn to_internal_dist_sq(dist: f64) -> f64 {
        let angle = (dist / EARTH_RADIUS_METERS).min(std::f64::consts::PI);
        let chord = 2.0 * (angle / 2.0).sin();
        chord * chord
    }

    fn from_internal_dist_sq(dist_sq: f64) -> f64 {
        let half_chord = (dist_sq.sqrt() / 2.0).min(1.0);
        2.0 * half_chord.asin() * EARTH_RADIUS_METERS
    }

    fn dist_sq(p: &SphericalPoint, q: &SphericalPoint) -> f64 {
        (p.xyz.0 - q.xyz.0).norm2()
    }

    // The closest point of the great circle is p's projection on it, and moving
    // away from it along the circle only gets further, so the closest point of the
    // segment is the projection if it's on the segment, else an endpoint. When p is
    // on the circle's axis every point of the circle is equally far.
    fn dist_sq_to_segment<S: Segment<P = SphericalPoint>>(p: &SphericalPoint, seg: &S) -> f64 {
        let endpoints = Self::dist_sq(p, seg.p1()).min(Self::dist_sq(p, seg.p2()));
        let n = unit_normal(seg);
        let off_axis = p.xyz.0 - n * n.dot(&p.xyz.0);
        if off_axis.norm() < EPS || !(0.0..=1.0).contains(&Self::projection_t(seg, p)) {
            return endpoints;
        }
        (p.xyz.0 - off_axis.normalize()).norm2().min(endpoints)
    }

    fn check_segment(a: &SphericalPoint, b: &SphericalPoint) -> Result<(), GeoError> {
        if a.is_equal(b) {
            return Err(GeoError::DegenerateSegment);
        }
        if (a.xyz.0 + b.xyz.0).norm() < EPS {
            return Err(GeoError::AntipodalSegment);
        }
        Ok(())
    }

    fn is_collinear<S: Segment<P = SphericalPoint>>(seg: &S, p: &SphericalPoint) -> bool {
        unit_normal(seg).dot(&p.xyz.0).abs() < EPS
    }

    // Signed, so that points before p1 get a negative t.
    fn projection_t<S: Segment<P = SphericalPoint>>(seg: &S, p: &SphericalPoint) -> f64 {
        let a = &seg.p1().xyz.0;
        let angle_p = a
            .cross(&p.xyz.0)
            .dot(&unit_normal(seg))
            .atan2(a.dot(&p.xyz.0));
        angle_p / a.angle(&seg.p2().xyz.0).rad()
    }

    fn midpoint<S: Segment<P = SphericalPoint>>(seg: &S) -> SphericalPoint {
        let v = (seg.p1().xyz.0 + seg.p2().xyz.0).normalize();
        SphericalPoint::from_xyz(S2Point(v))
    }

    // The great circles meet at two antipodal points, only the one on a's side of
    // the sphere (towards a's midpoint) can be on a. the clockwise flag indicates
    // if b's turn wrt a, at the crossing, is cw.
    fn intersection<A: Segment<P = SphericalPoint>, B: Segment<P = SphericalPoint>>(
        a: &A,
        b: &B,
    ) -> Option<SegmentIntersection> {
        let x = unit_normal(a).cross(&unit_normal(b));
        if x.norm() < EPS {
            return None;
        }
        let clockwise = x.dot(&(a.p1().xyz.0 + a.p2().xyz.0)) < 0.0;
        let x = if clockwise { x * -1.0 } else { x };
        let x = SphericalPoint::from_xyz(S2Point(x.normalize()));
        Some(SegmentIntersection {
            t: Self::projection_t(a, &x),
            u: Self::projection_t(b, &x),
            clockwise,
        })
    }

    fn same_direction<A: Segment<P = SphericalPoint>, B: Segment<P = SphericalPoint>>(
        a: &A,
        b: &B,
    ) -> bool {
        unit_normal(a).dot(&unit_normal(b)) > 0.0
    }

    // The tangent at `at` of the great circle from `from` to `to`.
    fn dir_at(from: &SphericalPoint, to: &SphericalPoint, at: &SphericalPoint) -> Vector {
        from.xyz.cross(&to.xyz).0.cross(&at.xyz.0).normalize()
    }

    // Counterclockwise as seen from outside the sphere.
    fn turn(at: &SphericalPoint, d1: &Vector, d2: &Vector) -> f64 {
        at.xyz.0.dot(&d1.cross(d2)).atan2(d1.dot(d2))
    }

    // Rings are taken as oriented (shells ccw, holes cw), as on the sphere
    // either side of a ring can be the interior.
    fn interior_on_left(_ring: &[SphericalPoint], _is_exterior: bool) -> bool {
        true
    }

    fn ray_crosses_edge(p: &SphericalPoint, edge: &SphericalEdge, next: &SphericalEdge) -> bool {
        let (p1, p2) = (edge.p1(), edge.p2());
        // The ray stops just short of the north pole, whatever happens at the pole
        // is accounted for by the reference's status.
        if p2.y == POLE_LAT {
            return false;
        }
        // Check the owned end vertex first: for a meridian edge sitting on the ray
        // (p1.x == p2.x == p.x) this resolves its p2 vertex via the wedge test.
        if p.x == p2.x {
            if p2.y <= p.y {
                return false;
            }
            return crossing_at_vertex::<Spherical>(edge, next, p, p2).is_some();
        }
        if p.x == p1.x {
            return false; // first endpoint: owned by the previous edge
        }
        lat_intercept(p1, p2, p).is_some_and(|lat| lat > p.y)
    }

    fn ray_crossing_distance(p: &SphericalPoint, edge: &SphericalEdge) -> Option<f64> {
        lat_intercept(edge.p1(), edge.p2(), p).map(|lat| lat - p.y)
    }

    // Rays go north along p's meridian, so b must share a's longitude band and
    // not be entirely south of a.
    fn ray_candidate(a: &IndexBox, b: &IndexBox) -> bool {
        a.min_x <= b.max_x + EPS && a.max_x >= b.min_x - EPS && b.max_y >= a.min_y - EPS
    }

    // From the haversine formula, for points at latitudes φ1, φ2 and a longitude
    // difference Δλ: hav(θ) = hav(φ1 - φ2) + cos(φ1) cos(φ2) hav(Δλ). Each term is
    // smallest with the smallest latitude and longitude gaps between the boxes, and
    // the latitude furthest from the equator. The squared chord is 4 hav(θ).
    fn box_min_dist_sq(a: &IndexBox, b: &IndexBox) -> f64 {
        let hav = |deg: f64| (deg.to_radians() / 2.0).sin().powi(2);
        let lat_gap = (a.min_y - b.max_y).max(b.min_y - a.max_y).max(0.0);
        let lon_gap = if a.min_x <= b.max_x && b.min_x <= a.max_x {
            0.0
        } else {
            let (left, right) = if a.max_x < b.min_x { (a, b) } else { (b, a) };
            (right.min_x - left.max_x).min(360.0 - (right.max_x - left.min_x))
        };
        let max_abs_lat = [a.min_y, a.max_y, b.min_y, b.max_y]
            .iter()
            .fold(0.0_f64, |acc, lat| acc.max(lat.abs()));
        let cos_sq = max_abs_lat.to_radians().cos().powi(2);
        4.0 * (hav(lat_gap) + cos_sq * hav(lon_gap))
    }

    fn reference_status(
        edges: &[SphericalEdge],
        poly_ids: &[usize],
        holes_start: usize,
    ) -> RefStatus {
        let mut encloses = vec![false; poly_ids.last().map_or(0, |&poly| poly + 1)];
        let mut pole_rings = Vec::new();
        let mut poles_differ = false;
        for ring in Ring::all(edges, poly_ids, holes_start) {
            if ring.pole_edge.is_some() {
                pole_rings.push(ring);
            } else {
                encloses[ring.poly_id] ^= ring.encloses_north_pole(edges);
                poles_differ ^= ring.separates_poles;
            }
        }
        RefStatus {
            ref_poly: encloses.iter().position(|&inside| inside),
            pole_rings,
            poles_differ,
        }
    }

    fn ring_encloses_ray_end(edges: &[SphericalEdge], ring: &Ring, p: &SphericalPoint) -> bool {
        ring.encloses_ray_end(edges, p)
    }

    // The edges' bboxes miss a pole that's inside the geometry (not on its
    // boundary, where they already reach it). The north pole is the reference. The
    // south pole's status follows from the north pole's and the rings separating
    // them, unless a ring goes through the north pole (where the north pole has no
    // status), then it's cast against like any point, its ray running up the whole
    // lon 0 meridian.
    fn poly_bbox(edges: &[SphericalEdge], edges_box: IndexBox, ref_status: &RefStatus) -> IndexBox {
        let mut bbox = edges_box;
        let extend = |bbox: &mut IndexBox, lat: f64| {
            bbox.min_x = -180.0;
            bbox.max_x = 180.0;
            bbox.min_y = bbox.min_y.min(lat);
            bbox.max_y = bbox.max_y.max(lat);
        };
        let north_inside = ref_status.ref_poly.is_some();
        if bbox.max_y < POLE_LAT && north_inside {
            extend(&mut bbox, POLE_LAT);
        }
        if bbox.min_y > -POLE_LAT {
            let south_inside = if ref_status.pole_rings.is_empty() {
                north_inside != ref_status.poles_differ
            } else {
                let south = SphericalPoint::new(0.0, -POLE_LAT);
                let crossings = edges
                    .iter()
                    .filter(|edge| Spherical::ray_crosses_edge(&south, edge, &edges[edge.next()]))
                    .count();
                (crossings % 2 == 1) != ref_status.inside_for::<Spherical>(edges, &south)
            };
            if south_inside {
                extend(&mut bbox, -POLE_LAT);
            }
        }
        bbox
    }
}

//************************************************************
// Helper functions for the spatial predicate checks, shared by the kernels.
//************************************************************
pub(crate) fn point_on_segment_t<K: GeoKernel, S: Segment<P = K::Point>>(
    p: &K::Point,
    seg: &S,
) -> Option<f64> {
    if !K::is_collinear(seg, p) {
        return None;
    }
    let t = K::projection_t(seg, p);
    if (-EPS..=(1.0 + EPS)).contains(&t) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

fn segments_collinear<K: GeoKernel, A: Segment<P = K::Point>, B: Segment<P = K::Point>>(
    a: &A,
    b: &B,
) -> bool {
    K::is_collinear(a, b.p1()) && K::is_collinear(a, b.p2())
}

// The portion of left (in left's parameter, clamped to [0, 1]) covered by
// right. Returns None when the two aren't collinear or the clamped overlap
// is empty.
pub(crate) fn covered_range<K: GeoKernel, A: Segment<P = K::Point>, B: Segment<P = K::Point>>(
    left: &A,
    right: &B,
) -> Option<(f64, f64)> {
    if !segments_collinear::<K, _, _>(left, right) {
        return None;
    }
    let ta = K::projection_t(left, right.p1());
    let tb = K::projection_t(left, right.p2());
    let t_lo = ta.min(tb).max(0.0);
    let t_hi = ta.max(tb).min(1.0);
    (t_lo <= t_hi + EPS).then_some((t_lo, t_hi))
}

// If p coincides with an endpoint of seg, returns Some(is_boundary), whether
// that endpoint is a linestring boundary vertex. Returns None otherwise.
pub(crate) fn point_at_line_endpoint<L: LineSegment>(p: &L::P, seg: &L) -> Option<bool> {
    if p.is_equal(seg.p1()) {
        Some(seg.p1_boundary())
    } else if p.is_equal(seg.p2()) {
        Some(seg.p2_boundary())
    } else {
        None
    }
}

// The parameters (t on a, u on b) of the intersection of two non-parallel
// segments, when it lands within both segments ([0, 1] inclusive, so endpoints
// count). None if there are no intersections.
pub(crate) fn segment_intersection<
    K: GeoKernel,
    A: Segment<P = K::Point>,
    B: Segment<P = K::Point>,
>(
    a: &A,
    b: &B,
) -> Option<(f64, f64)> {
    let i = K::intersection(a, b)?;
    let within = |v: f64| (-EPS..=1.0 + EPS).contains(&v);
    (within(i.t) && within(i.u)).then_some((i.t, i.u))
}

pub(crate) fn segments_cross<K: GeoKernel, A: Segment<P = K::Point>, B: Segment<P = K::Point>>(
    a: &A,
    b: &B,
) -> bool {
    segment_intersection::<K, _, _>(a, b).is_some()
}

// For two collinear, overlapping edges, whether their polygon interiors are on
// the same side — i.e. the two polygons overlap along the shared boundary (an
// interior–interior contact), rather than being adjacent (interiors on opposite
// sides, a boundary-only touch). Only meaningful for a collinear overlapping pair.
pub(crate) fn interiors_same_side<K: GeoKernel>(a: &K::Edge, b: &K::Edge) -> bool {
    let same_dir = K::same_direction(a, b);
    (a.interior_on_left() == b.interior_on_left()) == same_dir
}

// Decides whether a ray/segment in the direction from `from` to `to` properly
// crosses the polygon boundary at edge's end vertex. Some(_) = a proper crossing
// (the bool is if the segment is entering); None = tangent / not a crossing.
pub(crate) fn crossing_at_vertex<K: GeoKernel>(
    edge: &K::Edge,
    next: &K::Edge,
    from: &K::Point,
    to: &K::Point,
) -> Option<bool> {
    let (seg_in_wedge, flipped_in_wedge) = K::wedge_membership(edge, next, from, to);
    (seg_in_wedge != flipped_in_wedge).then_some(seg_in_wedge)
}

// Whether a line passing through edge's end vertex in the direction from `from`
// to `to` lies inside the polygon there, i.e. both the ray and its opposite fall
// within the interior wedge.
pub(crate) fn segment_inside_at_vertex<K: GeoKernel>(
    edge: &K::Edge,
    next: &K::Edge,
    from: &K::Point,
    to: &K::Point,
) -> bool {
    let (seg_in_wedge, flipped_in_wedge) = K::wedge_membership(edge, next, from, to);
    seg_in_wedge && flipped_in_wedge
}

// The ray-cast / midpoint classification of one left segment against one poly
// edge (edge, whose interior wedge is formed with next). Returns (ray_crossed,
// inside_at_vertex, parity_valid):
// - ray_crossed, the ray cast from the segment's midpoint crosses edge;
// - inside_at_vertex, the midpoint sits on the edge's owned vertex and the
//   segment lies inside the polygon there;
// - parity_valid, false when the midpoint sits on the boundary (on the vertex
//   or on the edge body), since ray parity through a boundary point is unreliable.
pub(crate) fn midpoint_ray_check<K: GeoKernel, S: Segment<P = K::Point>>(
    left: &S,
    edge: &K::Edge,
    next: &K::Edge,
) -> (bool, bool, bool) {
    let mid = K::midpoint(left);

    if mid.is_equal(edge.p2()) {
        let inside = segment_inside_at_vertex::<K>(edge, next, left.p1(), left.p2());
        return (false, inside, false);
    }
    if point_on_segment_t::<K, _>(&mid, edge).is_some() {
        return (false, false, false);
    }
    (K::ray_crosses_edge(&mid, edge, next), false, true)
}

// Classifies the intersection of one left segment with one poly edge (edge, whose
// interior wedge is formed with next). Returns (qualifying, signed_t):
// - qualifying, the segment touches the polygon interior here.
// - signed_t, a recorded crossing the crossing's parameter on the left segment, "+"
//   for entering, "−" for exiting. None when nothing is recorded.
// Collinear crossings can be triggered by either the edge or the next edge being
// collinear with the segment. Both need to be checked to detect entring the edge
// and leaving the edge.
pub(crate) fn segment_crossing_check<K: GeoKernel, S: Segment<P = K::Point>>(
    left: &S,
    edge: &K::Edge,
    next: &K::Edge,
) -> (bool, Option<f64>) {
    let (l1, l2) = (left.p1(), left.p2());
    let collinear_edge = segments_collinear::<K, _, _>(left, edge);
    let collinear_next = segments_collinear::<K, _, _>(left, next);

    if collinear_edge || collinear_next {
        // 180° pass-through (boundary continues straight through the vertex): ignore.
        if collinear_edge && collinear_next {
            return (false, None);
        }
        // The vertex must lie strictly within the left segment, else no crossing here
        // (the edge fully contains the segment, or they don't touch).
        let t = K::projection_t(left, edge.p2());
        if !(t > EPS && t < 1.0 - EPS) {
            return (false, None);
        }
        if K::is_reflex(edge, next) {
            // Peel into the interior: an interior contact, nothing to cancel.
            return (true, None);
        }
        // Convex peel into the exterior: record a signed crossing (not an interior
        // contact). Check if edge or next is the edge that's collinear with the
        // segment (either would lead to this), which tells you if you're crossing
        // the first endpoint or the second endpoint, then combine with the
        // directions, and you can tell if you entering or exiting.
        let exit = if collinear_edge {
            K::same_direction(left, edge)
        } else {
            !K::same_direction(left, next)
        };
        return (false, Some(if exit { -t } else { t }));
    }

    // Non-collinear: the intersection parameters, t on the left and u on the edge.
    let Some(i) = K::intersection(left, edge) else {
        return (false, None);
    };
    if !(i.t > EPS && i.t < 1.0 - EPS) {
        return (false, None);
    }
    if i.u > EPS && i.u < 1.0 - EPS {
        // Transversal crossing through the edge body: an interior contact, recorded
        // so it can cancel against another polygon's vertex touching this edge here.
        let is_entering = i.clockwise == edge.interior_on_left();
        return (true, Some(if is_entering { i.t } else { -i.t }));
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&i.u) {
        // Crossing at the owned vertex; exclude-start drops u ≈ 0 (owned by the
        // previous segment).
        if let Some(is_entering) = crossing_at_vertex::<K>(edge, next, l1, l2) {
            return (true, Some(if is_entering { i.t } else { -i.t }));
        }
    }
    (false, None)
}

// How one left polygon edge relates to one right polygon edge (edge, interior wedge
// formed with next), Like segment_crossing_check but for an Edge left side and with
// no signed-t bookkeeping (a single container, nothing to cancel). Returns (disqualifies,
// collinear_inside):
// - disqualifies: a transversal/owned-vertex crossing, or a collinear overlap with the
//   interiors on opposite sides (adjacent disjoint polys).
// - collinear_inside: collinear overlap with interiors on the same side.
// Both false = no contact here.
pub(crate) fn poly_edge_relation<K: GeoKernel>(
    left: &K::Edge,
    edge: &K::Edge,
    next: &K::Edge,
) -> (bool, bool) {
    let (l1, l2) = (left.p1(), left.p2());
    if segments_collinear::<K, _, _>(left, edge) {
        let t1 = K::projection_t(left, edge.p1());
        let t2 = K::projection_t(left, edge.p2());
        let lo = t1.min(t2).max(0.0);
        let hi = t1.max(t2).min(1.0);
        if hi - lo <= EPS {
            return (false, false);
        }
        let same_side = interiors_same_side::<K>(left, edge);
        return (!same_side, same_side);
    }

    // Non-collinear: the intersection parameters, t on the left and u on the edge.
    let Some(i) = K::intersection(left, edge) else {
        return (false, false);
    };
    if !(i.t > EPS && i.t < 1.0 - EPS) {
        return (false, false);
    }
    if i.u > EPS && i.u < 1.0 - EPS {
        return (true, false);
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&i.u)
        && crossing_at_vertex::<K>(edge, next, l1, l2).is_some()
    {
        // Enter/exit crossing at the edge's owned vertex (exclude-start drops u ≈ 0).
        return (true, false);
    }
    (false, false)
}

//*************************************************
// Natural hierarchical bbox index.
//
// A flat tree of aggregate bounding boxes over a geometry's ordered
// components (polygon edges or line segments). Only the internal levels
// are stored, leaf boxes are derived from the components on the fly during
// descent. geo_id_offset is added to leaf positions so a holes-only index
// still emits global edge indices.
//*************************************************
const INDEX_FANOUT: usize = 16;

// Bounding box covering every component in the slice.
fn union(items: &[impl Bboxable]) -> IndexBox {
    let mut it = items.iter();
    let first = it.next().expect("union of empty slice").bbox();
    it.fold(first, |acc, item| {
        let b = item.bbox();
        IndexBox {
            min_x: acc.min_x.min(b.min_x),
            min_y: acc.min_y.min(b.min_y),
            max_x: acc.max_x.max(b.max_x),
            max_y: acc.max_y.max(b.max_y),
        }
    })
}

pub(crate) struct NaturalIndex {
    boxes: Vec<IndexBox>,
    level_bases: Vec<usize>,
    geo_id_offset: usize,
}

impl NaturalIndex {
    fn from_components<C: Bboxable>(comps: &[C], geo_id_offset: usize) -> NaturalIndex {
        let n_leaves = comps.len();
        let mut boxes: Vec<IndexBox> = Vec::new();
        let mut level_bases: Vec<usize> = Vec::new();

        if n_leaves > 1 {
            // Finest internal level (starts at 0): union groups of up to
            // INDEX_FANOUT leaves.
            level_bases.push(0);
            let level1_count = n_leaves.div_ceil(INDEX_FANOUT);
            for p in 0..level1_count {
                let lo = p * INDEX_FANOUT;
                let hi = (lo + INDEX_FANOUT).min(n_leaves);
                boxes.push(union(&comps[lo..hi]));
            }

            // Higher levels: union groups of the previous (already-stored)
            // internal level.
            let mut level_start = 0;
            let mut level_len = level1_count;
            while level_len > 1 {
                let parents_start = boxes.len();
                level_bases.push(parents_start);
                let parent_count = level_len.div_ceil(INDEX_FANOUT);
                for p in 0..parent_count {
                    let lo = level_start + p * INDEX_FANOUT;
                    let hi = (lo + INDEX_FANOUT).min(level_start + level_len);
                    let parent = union(&boxes[lo..hi]);
                    boxes.push(parent);
                }
                level_start = parents_start;
                level_len = parent_count;
            }
        }

        NaturalIndex {
            boxes,
            level_bases,
            geo_id_offset,
        }
    }

    // A one-node index holding just the covering box of all components (empty when
    // there are none). Not traversable as a hierarchy; used only to expose an
    // overall bbox for the point variant.
    fn single_node<C: Bboxable>(comps: &[C]) -> NaturalIndex {
        let (boxes, level_bases) = if comps.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (vec![union(comps)], vec![0])
        };
        NaturalIndex {
            boxes,
            level_bases,
            geo_id_offset: 0,
        }
    }
}

//*************************************************
// An accumulator trait to handle index traversal when evaluating a predicate
// on 2 join candidates.

// Fuses candidate generation via index traversal with predicate evaluation:
// a descent calls update(li, ri) at each surviving leaf pair, with global component
// ids in the accumulator's (left, right) orientation. The accumulator captures
// the concrete component slices it needs up front, so update only indexes them.
//*************************************************
pub(crate) trait Accumulator {
    type K: GeoKernel;

    // Conservative box filter that drives index pruning: true means the descent
    // must look inside `b` for a possible match against `a`. Default is the
    // kernel's ray test, correct for ray-cast checks; crossing checks override
    // it with a symmetric bbox-overlap test.
    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        Self::K::ray_candidate(&a.bbox(), &b.bbox())
    }

    // Folds one candidate component pair into the running state.
    fn update(&mut self, li: usize, ri: usize);

    // Checked right after each `update`; true ends the descent early.
    fn ready(&self) -> bool;

    // The verdict, once the descent has finished (or early-exited).
    fn finish(self) -> bool;
}

// Single-descent kernel: iterate the left components, and for each descend the
// right index, folding every surviving leaf pair into the accumulator (left ids
// arrive grouped/contiguous). right_index is None for a Point side (no index),
// in which case we linear-scan the right leaves. The leaves themselves are passed
// separately because `NaturalIndex` only stores aggregate boxes, not the components.
fn single_descend<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    left: &[LC],
    right_leaves: &[RC],
    right_index: Option<&NaturalIndex>,
    acc: &mut A,
) {
    // Descend when there's an index with internal levels.
    if let Some(idx) = right_index {
        if !idx.boxes.is_empty() {
            let root_level = idx.level_bases.len() - 1;
            for (li, query) in left.iter().enumerate() {
                if descend_node::<LC, RC, A>(
                    query,
                    li,
                    right_leaves,
                    &idx.boxes,
                    &idx.level_bases,
                    idx.geo_id_offset,
                    root_level,
                    0,
                    acc,
                ) {
                    return;
                }
            }
            return;
        }
    }

    // Otherwise, scan the right leaves.
    let offset = right_index.map_or(0, |idx| idx.geo_id_offset);
    for (li, query) in left.iter().enumerate() {
        if scan_leaves::<LC, RC, A>(query, li, right_leaves, 0, right_leaves.len(), offset, acc) {
            return;
        }
    }
}

// Tests one left query against a contiguous run of right leaves; folds survivors.
// Returns true if the accumulator asked to stop.
fn scan_leaves<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    query: &LC,
    li: usize,
    right_leaves: &[RC],
    lo: usize,
    hi: usize,
    right_offset: usize,
    acc: &mut A,
) -> bool {
    for (i, leaf) in right_leaves[lo..hi].iter().enumerate() {
        if acc.prune(query, leaf) {
            acc.update(li, right_offset + lo + i);
            if acc.ready() {
                return true;
            }
        }
    }
    false
}

// Recursively descends one internal node of the right index for a single left
// query, pruning subtrees and bottoming out into `scan_leaves` at the finest
// level. Returns true to stop.
#[allow(clippy::too_many_arguments)]
fn descend_node<LC: Bboxable, RC: Bboxable, A: Accumulator>(
    query: &LC,
    li: usize,
    right_leaves: &[RC],
    right_boxes: &[IndexBox],
    bases: &[usize],
    right_offset: usize,
    level: usize,
    pos: usize,
    acc: &mut A,
) -> bool {
    if !acc.prune(query, &right_boxes[bases[level] + pos]) {
        return false;
    }
    let lo = pos * INDEX_FANOUT;
    if level == 0 {
        // Children are leaf components.
        let hi = (lo + INDEX_FANOUT).min(right_leaves.len());
        scan_leaves::<LC, RC, A>(query, li, right_leaves, lo, hi, right_offset, acc)
    } else {
        // Levels are contiguous in `boxes`, so the child level's node count is the gap between
        // its base and this level's base.
        let child_level_size = bases[level] - bases[level - 1];
        let hi = (lo + INDEX_FANOUT).min(child_level_size);
        for c in lo..hi {
            if descend_node::<LC, RC, A>(
                query,
                li,
                right_leaves,
                right_boxes,
                bases,
                right_offset,
                level - 1,
                c,
                acc,
            ) {
                return true;
            }
        }
        false
    }
}

// Dual-descent kernel: both sides indexed, descended in lockstep so they bottom
// out together. For order-independent checks. If either side has no internal
// levels a plain single descent is correct and keeps the left/right orientation,
// so we fall back to it.
fn run_dual<AC: Bboxable, BC: Bboxable, A: Accumulator>(
    a_leaves: &[AC],
    a_index: &NaturalIndex,
    b_leaves: &[BC],
    b_index: &NaturalIndex,
    acc: &mut A,
) {
    if a_index.boxes.is_empty() || b_index.boxes.is_empty() {
        single_descend(a_leaves, b_leaves, Some(b_index), acc);
        return;
    }
    dual_recurse::<AC, BC, A>(
        a_leaves,
        &a_index.boxes,
        &a_index.level_bases,
        a_index.geo_id_offset,
        b_leaves,
        &b_index.boxes,
        &b_index.level_bases,
        b_index.geo_id_offset,
        a_index.level_bases.len() - 1,
        0,
        b_index.level_bases.len() - 1,
        0,
        acc,
    );
}

// level/pos count internal levels with 0 = the leaf-parent level (its children
// are leaves). More levels left = higher level. Prune the node pair, then both
// at the leaf-parent level:
// - pair leaves via `scan_leaves`;
// - equal level above that → cartesian over child pairs;
// - unequal → descend only the deeper side so the levels re-sync.
// Returns true to stop (early exit).
#[allow(clippy::too_many_arguments)]
fn dual_recurse<AC: Bboxable, BC: Bboxable, A: Accumulator>(
    a_leaves: &[AC],
    a_boxes: &[IndexBox],
    a_bases: &[usize],
    a_offset: usize,
    b_leaves: &[BC],
    b_boxes: &[IndexBox],
    b_bases: &[usize],
    b_offset: usize,
    a_level: usize,
    a_pos: usize,
    b_level: usize,
    b_pos: usize,
    acc: &mut A,
) -> bool {
    let a_box = &a_boxes[a_bases[a_level] + a_pos];
    let b_box = &b_boxes[b_bases[b_level] + b_pos];
    if !acc.prune(a_box, b_box) {
        return false;
    }

    if a_level == 0 && b_level == 0 {
        // Both at the leaf-parent level: pair a's leaf range against b's leaf range.
        let a_lo = a_pos * INDEX_FANOUT;
        let a_hi = (a_lo + INDEX_FANOUT).min(a_leaves.len());
        let b_lo = b_pos * INDEX_FANOUT;
        let b_hi = (b_lo + INDEX_FANOUT).min(b_leaves.len());
        for (i, a_leaf) in a_leaves[a_lo..a_hi].iter().enumerate() {
            if scan_leaves::<AC, BC, A>(
                a_leaf,
                a_offset + a_lo + i,
                b_leaves,
                b_lo,
                b_hi,
                b_offset,
                acc,
            ) {
                return true;
            }
        }
        return false;
    }

    // Child position ranges for whichever side(s) we descend. A level's child count
    // is the contiguous gap to the next-finer level's base.
    let a_lo = a_pos * INDEX_FANOUT;
    let a_hi = (a_lo + INDEX_FANOUT).min(if a_level > 0 {
        a_bases[a_level] - a_bases[a_level - 1]
    } else {
        a_leaves.len()
    });
    let b_lo = b_pos * INDEX_FANOUT;
    let b_hi = (b_lo + INDEX_FANOUT).min(if b_level > 0 {
        b_bases[b_level] - b_bases[b_level - 1]
    } else {
        b_leaves.len()
    });

    if a_level > b_level {
        // Descend only A; keep B's node fixed until the levels re-sync.
        for ac in a_lo..a_hi {
            if dual_recurse::<AC, BC, A>(
                a_leaves,
                a_boxes,
                a_bases,
                a_offset,
                b_leaves,
                b_boxes,
                b_bases,
                b_offset,
                a_level - 1,
                ac,
                b_level,
                b_pos,
                acc,
            ) {
                return true;
            }
        }
    } else if b_level > a_level {
        // Descend only B.
        for bc in b_lo..b_hi {
            if dual_recurse::<AC, BC, A>(
                a_leaves,
                a_boxes,
                a_bases,
                a_offset,
                b_leaves,
                b_boxes,
                b_bases,
                b_offset,
                a_level,
                a_pos,
                b_level - 1,
                bc,
                acc,
            ) {
                return true;
            }
        }
    } else {
        // Equal level (> 0): descend both — cartesian over child pairs.
        for ac in a_lo..a_hi {
            for bc in b_lo..b_hi {
                if dual_recurse::<AC, BC, A>(
                    a_leaves,
                    a_boxes,
                    a_bases,
                    a_offset,
                    b_leaves,
                    b_boxes,
                    b_bases,
                    b_offset,
                    a_level - 1,
                    ac,
                    b_level - 1,
                    bc,
                    acc,
                ) {
                    return true;
                }
            }
        }
    }
    false
}

//*************************************************
// A represenation of a geometry as its components
//*************************************************
pub(crate) enum JoinableGeo<K: GeoKernel> {
    Point {
        points: Vec<K::Point>,

        // A one-node index holding just the covering bbox of the points.
        // It exists only so a point geo can expose an overall bbox.
        index: NaturalIndex,
    },
    Line {
        lines: Vec<K::Line>,

        // Index in `lines` of each linestring's first segment (segments of a
        // linestring are contiguous). Needed to tell disconnected linestrings
        // apart, since consecutive ones may share an endpoint.
        line_starts: Vec<usize>,
        index: NaturalIndex,
    },
    Poly {
        edges: Vec<K::Edge>,
        poly_ids: Vec<usize>,

        ref_status: RefStatus,

        // Covers everything the polys cover, which the index's root doesn't
        // necessarily do. None when there are no edges.
        bbox: Option<IndexBox>,

        // Index over all edges (exteriors + holes), geo_id_offset 0.
        index: NaturalIndex,

        // Index over hole edges only, geo_id_offset = the first
        // hole edge's global index.
        holes_index: NaturalIndex,
    },
}

impl<K: GeoKernel> JoinableGeo<K> {
    pub(crate) fn from_wkb(bytes: &[u8]) -> Result<Self, GeoError> {
        let wkb = Wkb::try_new(bytes).map_err(|e| GeoError::MalformedWkb(e.to_string()))?;
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let mut points = Vec::new();
                if let Some(coord) = p.coord() {
                    points.push(K::Point::new(coord.x(), coord.y()));
                }
                let index = NaturalIndex::single_node(&points);
                Ok(JoinableGeo::Point { points, index })
            }
            GeometryType::MultiPoint(mp) => {
                let mut points = Vec::new();
                for p in mp.points() {
                    if let Some(coord) = p.coord() {
                        points.push(K::Point::new(coord.x(), coord.y()));
                    }
                }
                let index = NaturalIndex::single_node(&points);
                Ok(JoinableGeo::Point { points, index })
            }
            GeometryType::LineString(ls) => build_line_geo(std::iter::once(ls)),
            GeometryType::MultiLineString(mls) => build_line_geo(mls.line_strings()),
            GeometryType::Polygon(poly) => build_poly_geo(std::iter::once(poly)),
            GeometryType::MultiPolygon(mp) => build_poly_geo(mp.polygons()),
            _ => todo!("from_wkb: unsupported geometry type"),
        }
    }
}

//*************************************************
// Predicate-facing traversal dispatch. The caller picks the method by what
// the check needs, fold_single when it needs self's components grouped
// (containment/parity), fold_dual for order-independent "any hit" checks.
// The accumulator (built by the caller for these geo types) is fed
// update(self_id, other_id).
//*************************************************
impl<K: GeoKernel> JoinableGeo<K> {
    // Grouped descent: iterate self's components, descend (or scan) other.
    // self's ids arrive contiguous, so accumulating checks can finalize per
    // group and early-exit.
    pub(crate) fn fold_for_grouped_check(
        &self,
        other: &JoinableGeo<K>,
        acc: &mut impl Accumulator<K = K>,
    ) {
        match self {
            JoinableGeo::Point { points, .. } => single_into(points, other, acc),
            JoinableGeo::Line { lines, .. } => single_into(lines, other, acc),
            JoinableGeo::Poly { edges, .. } => single_into(edges, other, acc),
        }
    }

    pub(crate) fn bounding_box(&self) -> Option<IndexBox> {
        match self {
            JoinableGeo::Point { index, .. } => index.boxes.last().copied(),
            JoinableGeo::Line { lines, index, .. } => index
                .boxes
                .last()
                .copied()
                .or_else(|| (!lines.is_empty()).then(|| union(lines))),
            JoinableGeo::Poly { bbox, .. } => *bbox,
        }
    }

    // Whether this geo's overall bbox overlaps another's (false when either
    // side is empty).
    pub(crate) fn top_bbox_overlaps(&self, other: &JoinableGeo<K>) -> bool {
        match (self.bounding_box(), other.bounding_box()) {
            (Some(a), Some(b)) => a.box_overlap(&b),
            _ => false,
        }
    }

    // Order-independent descent: lockstep dual descent when both sides are indexed,
    // otherwise a single descent (iterate self, descend/scan other), same traversal a
    // fold_single call would produce.
    pub(crate) fn fold_for_unordered_check(
        &self,
        other: &JoinableGeo<K>,
        acc: &mut impl Accumulator<K = K>,
    ) {
        match self {
            JoinableGeo::Point { points, .. } => dual_into(points, None, other, acc),
            JoinableGeo::Line { lines, index, .. } => dual_into(lines, Some(index), other, acc),
            JoinableGeo::Poly { edges, index, .. } => dual_into(edges, Some(index), other, acc),
        }
    }

    // Index into this poly's `edges` where the rings that can bound a part of its complement
    // begin: the holes (exterior rings are stored first), or every ring when shells can too
    // (K::SHELLS_BOUND_COMPLEMENT). `fold_complement_rings_into` delivers those edges to an
    // accumulator with ids relative to this offset, so the accumulator adds it to recover the edge
    // in the full vector. Zero for non-poly geometries.
    pub(crate) fn complement_rings_start(&self) -> usize {
        match self {
            JoinableGeo::Poly { holes_index, .. } if !K::SHELLS_BOUND_COMPLEMENT => {
                holes_index.geo_id_offset
            }
            _ => 0,
        }
    }

    // Grouped descent over only this poly's edges from complement_rings_start (an empty slice
    // when there are none, the descent then being a no-op) as the query side, descending other.
    // Poly-only: the caller (poly×poly within) is the sole user.
    pub(crate) fn fold_complement_rings_into(
        &self,
        other: &JoinableGeo<K>,
        acc: &mut impl Accumulator<K = K>,
    ) {
        match self {
            JoinableGeo::Poly { edges, .. } => {
                single_into(&edges[self.complement_rings_start()..], other, acc)
            }
            _ => unreachable!("fold_complement_rings_into is poly-only"),
        }
    }

    // Dumps the geometry as a list of vertex paths:
    // - Point: one single-vertex path per point.
    // - Line: one open path per linestring.
    // - Poly: one open ring per shell/hole (first vertex not repeated), oriented
    //   with the interior on the left, so shells are CCW and holes CW. Ring
    //   roles are carried by orientation alone, so all rings of all polys are
    //   flattened into one list.
    pub(crate) fn dump_paths(&self) -> Vec<Vec<[f64; 2]>> {
        match self {
            JoinableGeo::Point { points, .. } => points.iter().map(|p| vec![p.xy()]).collect(),
            JoinableGeo::Line {
                lines, line_starts, ..
            } => line_starts
                .iter()
                .enumerate()
                .map(|(k, &start)| {
                    let end = line_starts.get(k + 1).copied().unwrap_or(lines.len());
                    let segs = &lines[start..end];
                    let mut path: Vec<[f64; 2]> = segs.iter().map(|s| s.p1().xy()).collect();
                    if let Some(last) = segs.last() {
                        path.push(last.p2().xy());
                    }
                    path
                })
                .collect(),
            JoinableGeo::Poly { edges, .. } => {
                let mut rings = Vec::new();
                let mut start = 0;
                while start < edges.len() {
                    // Rings are contiguous; advance to the ring's last edge,
                    // the one whose `next` wraps back to the ring's first edge.
                    let mut end = start;
                    while edges[end].next() == end + 1 {
                        end += 1;
                    }
                    let mut ring: Vec<[f64; 2]> =
                        edges[start..=end].iter().map(|e| e.p1().xy()).collect();
                    if !edges[start].interior_on_left() {
                        ring.reverse();
                    }
                    rings.push(ring);
                    start = end + 1;
                }
                rings
            }
        }
    }
}

// Resolves other for a single descent: descend its index or scan it.
fn single_into<K: GeoKernel, LC: Bboxable, A: Accumulator<K = K>>(
    left: &[LC],
    other: &JoinableGeo<K>,
    acc: &mut A,
) {
    match other {
        JoinableGeo::Point { points, .. } => single_descend(left, points, None, acc),
        JoinableGeo::Line { lines, index, .. } => single_descend(left, lines, Some(index), acc),
        JoinableGeo::Poly { edges, index, .. } => single_descend(left, edges, Some(index), acc),
    }
}

// Resolves other for an order-independent descent: dual descent when both sides are indexed,
// else fall back to a single descent (iterating left).
fn dual_into<K: GeoKernel, LC: Bboxable, A: Accumulator<K = K>>(
    left: &[LC],
    left_index: Option<&NaturalIndex>,
    other: &JoinableGeo<K>,
    acc: &mut A,
) {
    match other {
        JoinableGeo::Point { points, .. } => single_descend(left, points, None, acc),
        JoinableGeo::Line { lines, index, .. } => match left_index {
            Some(li) => run_dual(left, li, lines, index, acc),
            None => single_descend(left, lines, Some(index), acc),
        },
        JoinableGeo::Poly { edges, index, .. } => match left_index {
            Some(li) => run_dual(left, li, edges, index, acc),
            None => single_descend(left, edges, Some(index), acc),
        },
    }
}

fn build_line_geo<K: GeoKernel, LS: LineStringTrait<T = f64>>(
    linestrings: impl Iterator<Item = LS>,
) -> Result<JoinableGeo<K>, GeoError> {
    let mut lines: Vec<K::Line> = Vec::new();
    let mut line_starts: Vec<usize> = Vec::new();

    for ls in linestrings {
        line_starts.push(lines.len());
        // Boundary flags are dummies here — the real mod-2 values are
        // computed by the self-join below.
        for (c1, c2) in ls
            .coords()
            .map(|c| K::Point::new(c.x(), c.y()))
            .tuple_windows::<(_, _)>()
        {
            K::check_segment(&c1, &c2)?;
            lines.push(K::Line::new(c1, c2, false, false));
        }
    }

    let index = NaturalIndex::from_components(&lines, 0);

    // Compute the mod-2 endpoint boundary flags via a cheap, faithful
    //  self-join over the segment endpoints as points. Each segment contributes
    // its two endpoints, laid out so endpoint 2*i is segment i's p1 and `2*i + 1`
    // is its p2; the tag is implicit in that layout. A flags entry is true
    // (boundary) unless an odd number of coincident endpoints cancel it.
    let mut endpoints: Vec<K::Point> = Vec::with_capacity(lines.len() * 2);
    for seg in &lines {
        endpoints.push(*seg.p1());
        endpoints.push(*seg.p2());
    }
    let flags = {
        let point_index = NaturalIndex::from_components(&endpoints, 0);
        let mut acc = PointCoincidence::<K>::new(&endpoints);
        run_dual(&endpoints, &point_index, &endpoints, &point_index, &mut acc);
        acc.flags
    };
    for (i, seg) in lines.iter_mut().enumerate() {
        seg.set_boundaries(flags[2 * i], flags[2 * i + 1]);
    }

    Ok(JoinableGeo::Line {
        lines,
        line_starts,
        index,
    })
}

// Accumulator for the construction-time endpoint self-join that computes
// the mod-2 boundary flags. Never ready (full traversal); `finish` is unused.
struct PointCoincidence<'a, K: GeoKernel> {
    points: &'a [K::Point],
    flags: Vec<bool>,
}

impl<'a, K: GeoKernel> PointCoincidence<'a, K> {
    fn new(points: &'a [K::Point]) -> Self {
        PointCoincidence {
            points,
            flags: vec![true; points.len()],
        }
    }
}

impl<K: GeoKernel> Accumulator for PointCoincidence<'_, K> {
    type K = K;

    // Symmetric EPS-slack bbox-overlap for the descent.
    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        // A point never matches itself. The symmetric dual traversal hands
        // each point its turn as `li` against every coincident neighbor, so
        // flipping only `flags[li]` flips each endpoint exactly once per
        // coincident neighbor.
        if li == ri {
            return;
        }
        if self.points[li].is_equal(&self.points[ri]) {
            self.flags[li] ^= true;
        }
    }

    fn ready(&self) -> bool {
        false
    }

    fn finish(self) -> bool {
        false
    }
}

// Handles one ring from one poly, appending its directed edges. Each edge records
// the index of its in-ring successor, patched at the ring end to wrap back to the
// ring's first edge.
fn process_ring<K: GeoKernel>(
    coords: impl Iterator<Item = impl CoordTrait<T = f64>>,
    is_exterior: bool,
    edges: &mut Vec<K::Edge>,
    poly_ids: &mut Vec<usize>,
    poly_idx: usize,
) -> Result<(), GeoError> {
    let ring_start = edges.len();
    let ring: Vec<K::Point> = coords.map(|c| K::Point::new(c.x(), c.y())).collect();

    for (a, b) in ring.iter().tuple_windows::<(_, _)>() {
        K::check_segment(a, b)?;
        let idx = edges.len();
        edges.push(K::Edge::new(*a, *b, idx + 1, false));
        poly_ids.push(poly_idx);
    }

    // A closed ring (first vertex == last vertex) produces at least one edge
    // ending where the first edge starts. No edges, or a mismatched end, means
    // it isn't a valid closed ring.
    if edges.len() == ring_start || !edges[edges.len() - 1].p2().is_equal(edges[ring_start].p1()) {
        return Err(GeoError::UnclosedRing);
    }
    let last = edges.len() - 1;
    edges[last].set_next(ring_start);

    let interior_on_left = K::interior_on_left(&ring, is_exterior);
    for edge in &mut edges[ring_start..] {
        edge.set_interior_on_left(interior_on_left);
    }
    Ok(())
}

fn build_poly_geo<K: GeoKernel, P: PolygonTrait<T = f64>>(
    polygons: impl Iterator<Item = P>,
) -> Result<JoinableGeo<K>, GeoError> {
    let mut edges: Vec<K::Edge> = Vec::new();
    let mut poly_ids: Vec<usize> = Vec::new();
    let polys: Vec<P> = polygons.collect();

    // Exterior rings first, then holes, so [0..holes_start) are exterior
    // edges and [holes_start..) are hole edges, addressable as one global
    // edge vector.
    for (poly_idx, poly) in polys.iter().enumerate() {
        let Some(exterior) = poly.exterior() else {
            continue;
        };
        process_ring::<K>(exterior.coords(), true, &mut edges, &mut poly_ids, poly_idx)?;
    }
    let holes_start = edges.len();
    for (poly_idx, poly) in polys.iter().enumerate() {
        for hole in poly.interiors() {
            process_ring::<K>(hole.coords(), false, &mut edges, &mut poly_ids, poly_idx)?;
        }
    }

    let ref_status = K::reference_status(&edges, &poly_ids, holes_start);
    let index = NaturalIndex::from_components(&edges, 0);
    let holes_index = NaturalIndex::from_components(&edges[holes_start..], holes_start);
    // A non-empty poly always has >= 3 edges, so its index is always built; an
    // empty index means an empty poly.
    let bbox = index
        .boxes
        .last()
        .map(|&edges_box| K::poly_bbox(&edges, edges_box, &ref_status));
    Ok(JoinableGeo::Poly {
        edges,
        poly_ids,
        ref_status,
        bbox,
        index,
        holes_index,
    })
}
