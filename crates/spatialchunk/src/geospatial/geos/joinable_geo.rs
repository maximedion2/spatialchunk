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
use wkb::reader::Wkb;

pub(crate) const EPS: f64 = 1e-10;

// Cheap geometry-validity errors caught at construction (not full
// DE-9IM validation, just the checks that prevent later misbehavior
// like divide-by-zero).
#[derive(Debug)]
pub(crate) enum GeoError {
    // The WKB bytes could not be parsed.
    MalformedWkb(String),
    // A segment/edge with identical endpoints (zero length).
    DegenerateSegment,
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
            GeoError::UnclosedRing => write!(f, "polygon ring is not closed"),
            GeoError::InvalidJoinSide => {
                write!(f, "JoinSide::None is not valid for a spatial join")
            }
        }
    }
}

impl std::error::Error for GeoError {}

//*************************************************
// Base component that all geometries are built from.
// all the basic checks are implemented here.
//*************************************************
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

#[derive(Copy, Clone, Debug)]
pub(crate) struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub(crate) fn x(&self) -> f64 {
        self.x
    }

    pub(crate) fn y(&self) -> f64 {
        self.y
    }

    pub(crate) fn is_equal(&self, other: &Point) -> bool {
        self.x == other.x && self.y == other.y
    }

    pub(crate) fn diff(&self, other: &Point) -> Dir {
        Dir {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    // Squared distance to another point.
    pub(crate) fn dist_sq(&self, other: &Point) -> f64 {
        let d = self.diff(other);
        d.dot(&d)
    }

    // Squared distance to a segment: project onto the segment, clamp the
    // parameter to the segment's extent, and measure to that closest point.
    pub(crate) fn dist_sq_to_segment(&self, seg: &impl SegmentTrait) -> f64 {
        let t = seg.projection_t(self).clamp(0.0, 1.0);
        let (p1, _) = seg.as_points();
        let dir = seg.dir();
        let closest = Point {
            x: p1.x + t * dir.x,
            y: p1.y + t * dir.y,
        };
        self.dist_sq(&closest)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LineSegment {
    p1: Point,
    p2: Point,
    p1_boundary: bool,
    p2_boundary: bool,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Edge {
    p1: Point,
    p2: Point,
    // Index (in the geometry's edge vector) of the next edge in this ring;
    // wraps to the ring's first edge at the ring end.
    next: usize,
    interior_on_left: bool,
}

pub(crate) trait AsPoints {
    fn as_points(&self) -> (&Point, &Point);
}

impl AsPoints for LineSegment {
    fn as_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }
}

impl AsPoints for Edge {
    fn as_points(&self) -> (&Point, &Point) {
        (&self.p1, &self.p2)
    }
}

// Geometric primitives shared by any segment-like component.
pub(crate) trait SegmentTrait: AsPoints + Sized {
    fn dir(&self) -> Dir {
        let (p1, p2) = self.as_points();
        p2.diff(p1)
    }

    fn norm_sq(&self) -> f64 {
        let d = self.dir();
        d.dot(&d)
    }

    fn cross(&self, other: &Dir) -> f64 {
        self.dir().cross(other)
    }

    fn dot(&self, other: &Dir) -> f64 {
        self.dir().dot(other)
    }

    fn is_collinear_with(&self, p: &Point) -> bool {
        let (p1, _) = self.as_points();
        let dp = p.diff(p1);
        let c = self.cross(&dp);
        c * c < EPS * EPS * self.norm_sq()
    }

    fn projection_t(&self, p: &Point) -> f64 {
        let (p1, _) = self.as_points();
        let dp = p.diff(p1);
        self.dot(&dp) / self.norm_sq()
    }

    // The x-coordinate where this segment reaches height p.y, or None when p.y is
    // outside the segment's y-span (the segment is not extended to infinity) or the
    // segment is horizontal (no single intercept).
    fn x_intercept_at_point(&self, p: &Point) -> Option<f64> {
        let (p1, p2) = self.as_points();
        let dy = p2.y - p1.y;
        if dy == 0.0 {
            return None;
        }
        let t = (p.y - p1.y) / dy;
        (0.0..=1.0).contains(&t).then_some(p1.x + t * (p2.x - p1.x))
    }

    fn midpoint(&self) -> Point {
        let (p1, p2) = self.as_points();
        Point {
            x: (p1.x + p2.x) / 2.0,
            y: (p1.y + p2.y) / 2.0,
        }
    }

    // Squared minimum distance to another segment, assuming the two do NOT
    // intersect (the caller must have already established that).
    fn dist_sq_to_disjoint(&self, other: &impl SegmentTrait) -> f64 {
        let (a1, a2) = self.as_points();
        let (b1, b2) = other.as_points();
        a1.dist_sq_to_segment(other)
            .min(a2.dist_sq_to_segment(other))
            .min(b1.dist_sq_to_segment(self))
            .min(b2.dist_sq_to_segment(self))
    }
}

impl<T: AsPoints> SegmentTrait for T {}

impl LineSegment {
    pub(crate) fn p1_boundary(&self) -> bool {
        self.p1_boundary
    }

    pub(crate) fn p2_boundary(&self) -> bool {
        self.p2_boundary
    }
}

impl Edge {
    pub(crate) fn interior_on_left(&self) -> bool {
        self.interior_on_left
    }

    // Index of the next edge in this ring (in the geometry's global edge vector).
    pub(crate) fn next(&self) -> usize {
        self.next
    }

    // For two collinear, overlapping edges, whether their polygon interiors are on
    // the same side — i.e. the two polygons overlap along the shared boundary (an
    // interior–interior contact), rather than being adjacent (interiors on opposite
    // sides, a boundary-only touch). Only meaningful for a collinear overlapping pair.
    pub(crate) fn interiors_same_side(&self, other: &Edge) -> bool {
        let same_dir = self.dir().dot(&other.dir()) > 0.0;
        (self.interior_on_left == other.interior_on_left) == same_dir
    }

    // Interior angle of the polygon at this edge's end vertex, from this edge's
    // direction and the next edge's direction.
    fn vertex_angle(&self, next: &Edge) -> f64 {
        let raw_turn = self.dir().turn_to(&next.dir());
        if self.interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        }
    }

    // Whether this edge's end vertex is reflex (interior angle > π).
    pub(crate) fn is_reflex(&self, next: &Edge) -> bool {
        self.vertex_angle(next) > std::f64::consts::PI
    }

    // Core wedge test at this edge's end vertex (interior wedge formed with `next`):
    // for a line through the vertex in direction `seg_dir`, whether the forward ray
    // (`seg_dir`) and its opposite each lie strictly inside the interior wedge.
    fn wedge_membership(&self, seg_dir: &Dir, next: &Edge) -> (bool, bool) {
        let angle = self.vertex_angle(next);
        let raw_turn = self.dir().turn_to(seg_dir);
        let seg_angle = if self.interior_on_left {
            std::f64::consts::PI - raw_turn
        } else {
            std::f64::consts::PI + raw_turn
        };
        let flipped = if seg_angle < std::f64::consts::PI {
            seg_angle + std::f64::consts::PI
        } else {
            seg_angle - std::f64::consts::PI
        };
        let seg_in_wedge = seg_angle > EPS && seg_angle < angle - EPS;
        let flipped_in_wedge = flipped > EPS && flipped < angle - EPS;
        (seg_in_wedge, flipped_in_wedge)
    }

    // Decides whether a ray/segment in direction seg_dir properly crosses the
    // polygon boundary at this edge's end vertex. Some(_) = a proper crossing
    // (the bool is if the segment is entering); None = tangent / not a crossing.
    pub(crate) fn crossing_at_vertex(&self, seg_dir: &Dir, next: &Edge) -> Option<bool> {
        let (seg_in_wedge, flipped_in_wedge) = self.wedge_membership(seg_dir, next);
        if seg_in_wedge != flipped_in_wedge {
            Some(seg_in_wedge)
        } else {
            None
        }
    }

    // Whether a line passing through this edge's end vertex in direction seg_dir
    // lies inside the polygon there, i.e. both the ray and its opposite fall within
    // the interior wedge.
    pub(crate) fn segment_inside_at_vertex(&self, seg_dir: &Dir, next: &Edge) -> bool {
        let (seg_in_wedge, flipped_in_wedge) = self.wedge_membership(seg_dir, next);
        seg_in_wedge && flipped_in_wedge
    }
}

//*************************************************
// Natural hierarchical bbox index.
//
// A flat tree of aggregate bounding boxes over a geometry's ordered components
// (polygon edges or line segments). Only the internal levels are stored; leaf boxes
// are derived from the components on the fly during descent. Levels are stored bottom-up:
// the finest internal level (parents of leaves) first, the root level last. The layout is
// implicit/positional, a node at (level, i) has children at (level+1, i*INDEX_FANOUT ..).
// geo_id_offset is added to leaf positions so a holes-only index (built over
// `edges[holes_start..]`) still emits global edge indices.
//*************************************************
const INDEX_FANOUT: usize = 16;

#[derive(Copy, Clone, Debug)]
pub(crate) struct IndexBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

// Implemented by both internal index nodes and leaf components (Edge,
// LineSegment), so index construction (parent = union of children) and
// descent are uniform and fully monomorphized (no `dyn`).
pub(crate) trait Bboxable {
    fn bbox(&self) -> IndexBox;

    // Rightward-ray descent test: `self` is the casting component/node, other a
    // candidate node. True when their y-bands overlap and `other` is not entirely
    // left of self, so a +x ray from self could reach it. Comparisons are slackened
    // by `EPS` so the prune is never tighter than the EPS-tolerant exact tests
    // downstream (a candidate within tolerance must not be pruned away).
    fn ray_candidate(&self, other: &impl Bboxable) -> bool {
        let a = self.bbox();
        let b = other.bbox();
        a.min_y <= b.max_y + EPS && a.max_y >= b.min_y - EPS && b.max_x >= a.min_x - EPS
    }

    // Symmetric bbox overlap (EPS-slackened). The prune for checks that don't
    // invovle ray casting.
    fn box_overlap(&self, other: &impl Bboxable) -> bool {
        let a = self.bbox();
        let b = other.bbox();
        a.min_x <= b.max_x + EPS
            && a.max_x >= b.min_x - EPS
            && a.min_y <= b.max_y + EPS
            && a.max_y >= b.min_y - EPS
    }

    // Squared minimum distance between the two boxes (0 when they overlap).
    fn box_min_dist_sq(&self, other: &impl Bboxable) -> f64 {
        let a = self.bbox();
        let b = other.bbox();
        let dx = (a.min_x - b.max_x).max(b.min_x - a.max_x).max(0.0);
        let dy = (a.min_y - b.max_y).max(b.min_y - a.max_y).max(0.0);
        dx * dx + dy * dy
    }
}

impl Bboxable for IndexBox {
    fn bbox(&self) -> IndexBox {
        *self
    }
}

impl Bboxable for Point {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.x,
            min_y: self.y,
            max_x: self.x,
            max_y: self.y,
        }
    }
}

impl Bboxable for LineSegment {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.p1.x.min(self.p2.x),
            min_y: self.p1.y.min(self.p2.y),
            max_x: self.p1.x.max(self.p2.x),
            max_y: self.p1.y.max(self.p2.y),
        }
    }
}

impl Bboxable for Edge {
    fn bbox(&self) -> IndexBox {
        IndexBox {
            min_x: self.p1.x.min(self.p2.x),
            min_y: self.p1.y.min(self.p2.y),
            max_x: self.p1.x.max(self.p2.x),
            max_y: self.p1.y.max(self.p2.y),
        }
    }
}

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
// A represenation of a geometry as its components
//*************************************************
pub(crate) enum JoinableGeo {
    Point {
        points: Vec<Point>,

        // A one-node index holding just the covering bbox of the points.
        // It exists only so a point geo can expose an overall bbox.
        index: NaturalIndex,
    },
    Line {
        lines: Vec<LineSegment>,

        // Index in `lines` of each linestring's first segment (segments of a
        // linestring are contiguous). Needed to tell disconnected linestrings
        // apart, since consecutive ones may share an endpoint.
        line_starts: Vec<usize>,
        index: NaturalIndex,
    },
    Poly {
        edges: Vec<Edge>,
        poly_ids: Vec<usize>,

        // Index over all edges (exteriors + holes), geo_id_offset 0.
        index: NaturalIndex,

        // Index over hole edges only, geo_id_offset = the first
        // hole edge's global index.
        holes_index: NaturalIndex,
    },
}

impl JoinableGeo {
    pub(crate) fn from_wkb(bytes: &[u8]) -> Result<Self, GeoError> {
        let wkb = Wkb::try_new(bytes).map_err(|e| GeoError::MalformedWkb(e.to_string()))?;
        match wkb.as_type() {
            GeometryType::Point(p) => {
                let mut points = Vec::new();
                if let Some(coord) = p.coord() {
                    points.push(Point {
                        x: coord.x(),
                        y: coord.y(),
                    });
                }
                let index = NaturalIndex::single_node(&points);
                Ok(JoinableGeo::Point { points, index })
            }
            GeometryType::MultiPoint(mp) => {
                let mut points = Vec::new();
                for p in mp.points() {
                    if let Some(coord) = p.coord() {
                        points.push(Point {
                            x: coord.x(),
                            y: coord.y(),
                        });
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

fn build_line_geo<LS: LineStringTrait<T = f64>>(
    linestrings: impl Iterator<Item = LS>,
) -> Result<JoinableGeo, GeoError> {
    let mut lines: Vec<LineSegment> = Vec::new();
    let mut line_starts: Vec<usize> = Vec::new();

    for ls in linestrings {
        line_starts.push(lines.len());
        // Boundary flags are dummies here — the real mod-2 values are
        // computed by the self-join below.
        for (c1, c2) in ls
            .coords()
            .map(|c| Point { x: c.x(), y: c.y() })
            .tuple_windows::<(_, _)>()
        {
            if c1.is_equal(&c2) {
                return Err(GeoError::DegenerateSegment);
            }
            lines.push(LineSegment {
                p1: c1,
                p2: c2,
                p1_boundary: false,
                p2_boundary: false,
            });
        }
    }

    let index = NaturalIndex::from_components(&lines, 0);

    // Compute the mod-2 endpoint boundary flags via a cheap, faithful
    //  self-join over the segment endpoints as points. Each segment contributes
    // its two endpoints, laid out so endpoint 2*i is segment i's p1 and `2*i + 1`
    // is its p2; the tag is implicit in that layout. A flags entry is true
    // (boundary) unless an odd number of coincident endpoints cancel it.
    let mut endpoints: Vec<Point> = Vec::with_capacity(lines.len() * 2);
    for seg in &lines {
        endpoints.push(seg.p1);
        endpoints.push(seg.p2);
    }
    let flags = {
        let point_index = NaturalIndex::from_components(&endpoints, 0);
        let mut acc = PointCoincidence::new(&endpoints);
        run_dual(&endpoints, &point_index, &endpoints, &point_index, &mut acc);
        acc.flags
    };
    for (i, seg) in lines.iter_mut().enumerate() {
        seg.p1_boundary = flags[2 * i];
        seg.p2_boundary = flags[2 * i + 1];
    }

    Ok(JoinableGeo::Line {
        lines,
        line_starts,
        index,
    })
}

// Accumulator for the construction-time endpoint self-join that computes
// the mod-2 boundary flags. Never ready (full traversal); `finish` is unused.
struct PointCoincidence<'a> {
    points: &'a [Point],
    flags: Vec<bool>,
}

impl<'a> PointCoincidence<'a> {
    fn new(points: &'a [Point]) -> Self {
        PointCoincidence {
            points,
            flags: vec![true; points.len()],
        }
    }
}

impl Accumulator for PointCoincidence<'_> {
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
fn process_ring(
    coords: impl Iterator<Item = impl CoordTrait<T = f64>>,
    is_exterior: bool,
    edges: &mut Vec<Edge>,
    poly_ids: &mut Vec<usize>,
    poly_idx: usize,
) -> Result<(), GeoError> {
    let ring_start = edges.len();
    let mut signed_area = 0.0f64;

    for (a, b) in coords
        .map(|c| Point { x: c.x(), y: c.y() })
        .tuple_windows::<(_, _)>()
    {
        if a.is_equal(&b) {
            return Err(GeoError::DegenerateSegment);
        }
        signed_area += a.x * b.y - b.x * a.y;
        let idx = edges.len();
        edges.push(Edge {
            p1: a,
            p2: b,
            next: idx + 1,
            interior_on_left: false,
        });
        poly_ids.push(poly_idx);
    }

    // A closed ring (first vertex == last vertex) produces at least one edge
    // ending where the first edge starts. No edges, or a mismatched end, means
    // it isn't a valid closed ring.
    if edges.len() == ring_start || !edges[edges.len() - 1].p2.is_equal(&edges[ring_start].p1) {
        return Err(GeoError::UnclosedRing);
    }
    let last = edges.len() - 1;
    edges[last].next = ring_start;

    // Determine ring orientation, then which side the polygon interior is on.
    let ring_is_ccw = signed_area > 0.0;
    let interior_on_left = if is_exterior {
        ring_is_ccw
    } else {
        !ring_is_ccw
    };
    for edge in &mut edges[ring_start..] {
        edge.interior_on_left = interior_on_left;
    }
    Ok(())
}

fn build_poly_geo<P: PolygonTrait<T = f64>>(
    polygons: impl Iterator<Item = P>,
) -> Result<JoinableGeo, GeoError> {
    let mut edges: Vec<Edge> = Vec::new();
    let mut poly_ids: Vec<usize> = Vec::new();
    let polys: Vec<P> = polygons.collect();

    // Exterior rings first, then holes, so [0..holes_start) are exterior
    // edges and [holes_start..) are hole edges, addressable as one global
    // edge vector.
    for (poly_idx, poly) in polys.iter().enumerate() {
        let Some(exterior) = poly.exterior() else {
            continue;
        };
        process_ring(exterior.coords(), true, &mut edges, &mut poly_ids, poly_idx)?;
    }
    let holes_start = edges.len();
    for (poly_idx, poly) in polys.iter().enumerate() {
        for hole in poly.interiors() {
            process_ring(hole.coords(), false, &mut edges, &mut poly_ids, poly_idx)?;
        }
    }

    let index = NaturalIndex::from_components(&edges, 0);
    let holes_index = NaturalIndex::from_components(&edges[holes_start..], holes_start);
    Ok(JoinableGeo::Poly {
        edges,
        poly_ids,
        index,
        holes_index,
    })
}

// Fuses candidate generation via index traversal with predicate evaluation:
// a descent calls update(li, ri) at each surviving leaf pair, with global component
// ids in the accumulator's (left, right) orientation. The accumulator captures
// the concrete component slices it needs up front, so update only indexes them,
// it never touches the `JoinableGeo` enum.
pub(crate) trait Accumulator {
    // Conservative box filter that drives index pruning: true means the descent
    // must look inside `b` for a possible match against `a`. Default is the
    // rightward-ray test, correct for ray-cast checks; crossing checks override
    // it with a symmetric bbox-overlap test.
    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.ray_candidate(b)
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
// Predicate-facing traversal dispatch. The caller picks the method by what
// the check needs, fold_single when it needs self's components grouped
// (containment/parity), fold_dual for order-independent "any hit" checks.
// The accumulator (built by the caller for these geo types) is fed
// update(self_id, other_id).
//*************************************************
impl JoinableGeo {
    // Grouped descent: iterate self's components, descend (or scan) other.
    // self's ids arrive contiguous, so accumulating checks can finalize per
    // group and early-exit.
    pub(crate) fn fold_for_grouped_check(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Point { points, .. } => single_into(points, other, acc),
            JoinableGeo::Line { lines, .. } => single_into(lines, other, acc),
            JoinableGeo::Poly { edges, .. } => single_into(edges, other, acc),
        }
    }

    fn bounding_box(&self) -> Option<IndexBox> {
        match self {
            JoinableGeo::Point { index, .. } => index.boxes.last().copied(),
            JoinableGeo::Line { lines, index, .. } => index
                .boxes
                .last()
                .copied()
                .or_else(|| (!lines.is_empty()).then(|| union(lines))),
            // A non-empty poly always has >= 3 edges, so its index is always built;
            // an empty index means an empty poly, for which None is correct.
            JoinableGeo::Poly { index, .. } => index.boxes.last().copied(),
        }
    }

    // Whether this geo's overall bbox overlaps another's (false when either
    // side is empty).
    pub(crate) fn top_bbox_overlaps(&self, other: &JoinableGeo) -> bool {
        match (self.bounding_box(), other.bounding_box()) {
            (Some(a), Some(b)) => a.box_overlap(&b),
            _ => false,
        }
    }

    // Order-independent descent: lockstep dual descent when both sides are indexed,
    // otherwise a single descent (iterate self, descend/scan other), same traversal a
    // fold_single call would produce.
    pub(crate) fn fold_for_unordered_check(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Point { points, .. } => dual_into(points, None, other, acc),
            JoinableGeo::Line { lines, index, .. } => dual_into(lines, Some(index), other, acc),
            JoinableGeo::Poly { edges, index, .. } => dual_into(edges, Some(index), other, acc),
        }
    }

    // Index into this poly's `edges` where the hole edges begin (exterior rings are stored first).
    // Equal to `edges.len()` when the poly has no holes. `fold_holes_into` delivers hole edges to an
    // accumulator with *hole-relative* left ids, so the accumulator adds this offset to recover the
    // edge in the full vector. Zero for non-poly geometries (which have no holes).
    pub(crate) fn holes_start(&self) -> usize {
        match self {
            JoinableGeo::Poly { holes_index, .. } => holes_index.geo_id_offset,
            _ => 0,
        }
    }

    // Grouped descent over only this poly's hole edges as the query side, descending other.
    // Holes occupy edges[holes_index.geo_id_offset..] (an empty slice when the poly has none, so
    // the descent is then a no-op). Poly-only: the caller (poly×poly within) is the sole user.
    pub(crate) fn fold_holes_into(&self, other: &JoinableGeo, acc: &mut impl Accumulator) {
        match self {
            JoinableGeo::Poly {
                edges, holes_index, ..
            } => single_into(&edges[holes_index.geo_id_offset..], other, acc),
            _ => unreachable!("fold_holes_into is poly-only"),
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
            JoinableGeo::Point { points, .. } => points.iter().map(|p| vec![[p.x, p.y]]).collect(),
            JoinableGeo::Line {
                lines, line_starts, ..
            } => line_starts
                .iter()
                .enumerate()
                .map(|(k, &start)| {
                    let end = line_starts.get(k + 1).copied().unwrap_or(lines.len());
                    let segs = &lines[start..end];
                    let mut path: Vec<[f64; 2]> = segs.iter().map(|s| [s.p1.x, s.p1.y]).collect();
                    if let Some(last) = segs.last() {
                        path.push([last.p2.x, last.p2.y]);
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
                    while edges[end].next == end + 1 {
                        end += 1;
                    }
                    let mut ring: Vec<[f64; 2]> = edges[start..=end]
                        .iter()
                        .map(|e| [e.p1.x, e.p1.y])
                        .collect();
                    if !edges[start].interior_on_left {
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
fn single_into<LC: Bboxable, A: Accumulator>(left: &[LC], other: &JoinableGeo, acc: &mut A) {
    match other {
        JoinableGeo::Point { points, .. } => single_descend(left, points, None, acc),
        JoinableGeo::Line { lines, index, .. } => single_descend(left, lines, Some(index), acc),
        JoinableGeo::Poly { edges, index, .. } => single_descend(left, edges, Some(index), acc),
    }
}

// Resolves other for an order-independent descent: dual descent when both sides are indexed,
// else fall back to a single descent (iterating left).
fn dual_into<LC: Bboxable, A: Accumulator>(
    left: &[LC],
    left_index: Option<&NaturalIndex>,
    other: &JoinableGeo,
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

//************************************************************
// Various helper functions for spatial predicate checks.
//************************************************************
pub(crate) fn point_on_segment_t(p: &Point, seg: &impl SegmentTrait) -> Option<f64> {
    if !seg.is_collinear_with(p) {
        return None;
    }
    let t = seg.projection_t(p);
    if (-EPS..=(1.0 + EPS)).contains(&t) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

// The portion of left (in left's parameter, clamped to [0, 1]) covered by
// right. Returns None when the two aren't collinear or the clamped overlap
// is empty.
pub(crate) fn covered_range(
    left: &impl SegmentTrait,
    right: &impl SegmentTrait,
) -> Option<(f64, f64)> {
    if !segments_collinear(left, right) {
        return None;
    }
    let (r1, r2) = right.as_points();
    let ta = left.projection_t(r1);
    let tb = left.projection_t(r2);
    let t_lo = ta.min(tb).max(0.0);
    let t_hi = ta.max(tb).min(1.0);
    (t_lo <= t_hi + EPS).then_some((t_lo, t_hi))
}
// If p coincides with an endpoint of seg, returns Some(is_boundary), whether
// that endpoint is a linestring boundary vertex. Returns None otherwise.

pub(crate) fn point_at_line_endpoint(p: &Point, seg: &LineSegment) -> Option<bool> {
    if p.is_equal(&seg.p1) {
        Some(seg.p1_boundary)
    } else if p.is_equal(&seg.p2) {
        Some(seg.p2_boundary)
    } else {
        None
    }
}

pub(crate) fn ray_crosses_edge(p: &Point, edge: &Edge, next: &Edge) -> bool {
    let (p1, p2) = edge.as_points();
    // Check the owned end vertex first: for a horizontal edge sitting on the ray
    // (p1.y == p2.y == p.y) this resolves its p2 vertex via the wedge test.
    if p.y == p2.y {
        if p2.x <= p.x {
            return false;
        }
        return edge.crossing_at_vertex(&p2.diff(p), next).is_some();
    }
    if p.y == p1.y {
        return false; // first endpoint: owned by the previous edge
    }
    // x_intercept_at_point returns None when p.y is outside the edge's y-span.
    edge.x_intercept_at_point(p).is_some_and(|xi| xi > p.x)
}

// The ray-cast / midpoint classification of one left segment against one poly
// edge (edge, whose interior wedge is formed with next). Returns (ray_crossed,
// inside_at_vertex, parity_valid):
// - ray_crossed, the rightward ray from the segment's midpoint crosses edge;
// - inside_at_vertex, the midpoint sits on the edge's owned vertex and the
//   segment lies inside the polygon there;
// - parity_valid, false when the midpoint sits on the boundary (on the vertex
//   or on the edge body), since ray parity through a boundary point is unreliable.
pub(crate) fn midpoint_ray_check(
    left: &impl SegmentTrait,
    edge: &Edge,
    next: &Edge,
) -> (bool, bool, bool) {
    let mid = left.midpoint();
    let (_, p2) = edge.as_points();

    if mid.is_equal(p2) {
        let inside = edge.segment_inside_at_vertex(&left.dir(), next);
        return (false, inside, false);
    }
    if point_on_segment_t(&mid, edge).is_some() {
        return (false, false, false);
    }
    (ray_crosses_edge(&mid, edge, next), false, true)
}

// Classifies the intersection of one left segment with one poly edge (edge, whose
// interior wedge is formed with next). Returns (qualifying, signed_t):
// - qualifying, the segment touches the polygon interior here.
// - signed_t, a recorded crossing the crossing's parameter on the left segment, "+"
//   for entering, "−" for exiting. None when nothing is recorded.
// Collinear crossings can be triggered by either the edge or the next edge being
// collinear with the segment. Both need to be checked to detect entring the edge
// and leaving the edge.
pub(crate) fn segment_crossing_check(
    left: &impl SegmentTrait,
    edge: &Edge,
    next: &Edge,
) -> (bool, Option<f64>) {
    let (e1, e2) = edge.as_points();
    let collinear_edge = left.is_collinear_with(e1) && left.is_collinear_with(e2);
    let (n1, n2) = next.as_points();
    let collinear_next = left.is_collinear_with(n1) && left.is_collinear_with(n2);

    if collinear_edge || collinear_next {
        // 180° pass-through (boundary continues straight through the vertex): ignore.
        if collinear_edge && collinear_next {
            return (false, None);
        }
        // The vertex must lie strictly within the left segment, else no crossing here
        // (the edge fully contains the segment, or they don't touch).
        let t = left.projection_t(e2);
        if !(t > EPS && t < 1.0 - EPS) {
            return (false, None);
        }
        if edge.is_reflex(next) {
            // Peel into the interior: an interior contact, nothing to cancel.
            return (true, None);
        }
        // Convex peel into the exterior: record a signed crossing (not an interior
        // contact). Check if edge or next is the edge that's collinear with the
        // segment (either would lead to this), which tells you if you're crossing
        // the first endpoint or the second endpoint, then combine with the dot
        // product of the directions, and you can tell if you entering or exiting.
        let exit = if collinear_edge {
            left.dir().dot(&edge.dir()) > 0.0
        } else {
            left.dir().dot(&next.dir()) < 0.0
        };
        return (false, Some(if exit { -t } else { t }));
    }

    // Non-collinear: solve the line intersection for t (on the left) and u
    // (on the edge).
    let cross = left.dir().cross(&edge.dir());
    if cross.abs() < EPS {
        return (false, None);
    }
    let (a1, _) = left.as_points();
    let f = e1.diff(a1); // edge.p1 − left.p1
    let t = -edge.dir().cross(&f) / cross;
    let u = -left.dir().cross(&f) / cross;
    if !(t > EPS && t < 1.0 - EPS) {
        return (false, None);
    }
    if u > EPS && u < 1.0 - EPS {
        // Transversal crossing through the edge body: an interior contact, recorded
        // so it can cancel against another polygon's vertex touching this edge here.
        let is_entering = (cross < 0.0) == edge.interior_on_left();
        return (true, Some(if is_entering { t } else { -t }));
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&u) {
        // Crossing at the owned vertex; exclude-start drops u ≈ 0 (owned by the
        // previous segment).
        if let Some(is_entering) = edge.crossing_at_vertex(&left.dir(), next) {
            return (true, Some(if is_entering { t } else { -t }));
        }
    }
    (false, None)
}

fn segments_collinear(a: &impl SegmentTrait, b: &impl SegmentTrait) -> bool {
    let (q1, q2) = b.as_points();
    a.is_collinear_with(q1) && a.is_collinear_with(q2)
}

// The parameters (t on a, u on b) of the intersection of two non-parallel
// segments, when it lands within both segments ([0, 1] inclusive, so endpoints
// count). None if there are no intersections.
pub(crate) fn segment_intersection(
    a: &impl SegmentTrait,
    b: &impl SegmentTrait,
) -> Option<(f64, f64)> {
    let cross = a.dir().cross(&b.dir());
    if cross.abs() < EPS {
        return None;
    }
    let (a1, _) = a.as_points();
    let (b1, _) = b.as_points();
    let f = b1.diff(a1);
    let t = -b.dir().cross(&f) / cross;
    let u = -a.dir().cross(&f) / cross;
    ((-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u)).then_some((t, u))
}

pub(crate) fn segments_cross(a: &impl SegmentTrait, b: &impl SegmentTrait) -> bool {
    segment_intersection(a, b).is_some()
}

// How one left polygon edge relates to one right polygon edge (edge, interior wedge
// formed with next), Like segment_crossing_check but for an Edge left side and with
// no signed-t bookkeeping (a single container, nothing to cancel). Returns (disqualifies,
// collinear_inside):
// - disqualifies: a transversal/owned-vertex crossing, or a collinear overlap with the
//   interiors on opposite sides (adjacent disjoint polys).
// - collinear_inside: collinear overlap with interiors on the same side.
// Both false = no contact here.
pub(crate) fn poly_edge_relation(left: &Edge, edge: &Edge, next: &Edge) -> (bool, bool) {
    let (e1, e2) = edge.as_points();
    if left.is_collinear_with(e1) && left.is_collinear_with(e2) {
        let t1 = left.projection_t(e1);
        let t2 = left.projection_t(e2);
        let lo = t1.min(t2).max(0.0);
        let hi = t1.max(t2).min(1.0);
        if hi - lo <= EPS {
            return (false, false);
        }
        // interior_on_left is relative to each edge's own direction, so a flip is
        // needed when the edges run opposite ways.
        let same_dir = left.dir().dot(&edge.dir()) > 0.0;
        let same_side = (left.interior_on_left() == edge.interior_on_left()) == same_dir;
        return (!same_side, same_side);
    }

    // Non-collinear: solve the line intersection for `t` (on the left) and `u` (on the edge).
    let cross = left.dir().cross(&edge.dir());
    if cross.abs() < EPS {
        return (false, false);
    }
    let (a1, _) = left.as_points();
    let f = e1.diff(a1);
    let t = -edge.dir().cross(&f) / cross;
    let u = -left.dir().cross(&f) / cross;
    if !(t > EPS && t < 1.0 - EPS) {
        return (false, false);
    }
    if u > EPS && u < 1.0 - EPS {
        return (true, false);
    }
    if (1.0 - EPS..=1.0 + EPS).contains(&u) && edge.crossing_at_vertex(&left.dir(), next).is_some()
    {
        // Enter/exit crossing at the edge's owned vertex (exclude-start drops u ≈ 0).
        return (true, false);
    }
    (false, false)
}
