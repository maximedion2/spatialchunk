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

use crate::geospatial::geos::joinable_geo::{
    covered_range, interiors_same_side, midpoint_ray_check, point_at_line_endpoint,
    point_on_segment_t, segment_crossing_check, segment_intersection, segments_cross, Accumulator,
    Bboxable, Edge, GeoKernel, JoinableGeo, LineSegment, Planar, PlanarEdge, PlanarLineSegment,
    PlanarPoint, Segment, EPS,
};

// *************************************************************
// st_touches(a, b) -> true if a and b share at least one point but their
// interiors do not intersect. Unlike st_intersects, this needs the
// boundary/interior distinction. Planar only: touching is exact boundary contact,
// which geographies' curved edges don't make reliable.
// *************************************************************
pub(crate) fn st_touches(a: &JoinableGeo<Planar>, b: &JoinableGeo<Planar>) -> bool {
    match (a, b) {
        // Points have empty boundary (interior = the point), so any contact
        // between two point sets is interior–interior. Touches is never true.
        (JoinableGeo::Point { .. }, JoinableGeo::Point { .. }) => false,

        (JoinableGeo::Point { points, .. }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointTouchesLine::new(points, lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }
        (JoinableGeo::Line { .. }, JoinableGeo::Point { .. }) => st_touches(b, a),

        (JoinableGeo::Point { points, .. }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = PointTouchesPoly::new(points, edges);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }
        (JoinableGeo::Poly { .. }, JoinableGeo::Point { .. }) => st_touches(b, a),

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            let mut acc = LineTouchesLine::new(l_lines, r_lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines, .. }, JoinableGeo::Poly { edges, .. }) => {
            let mut acc = LineTouchesPoly::new(lines, edges);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }
        (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => st_touches(b, a),

        (JoinableGeo::Poly { edges: a_edges, .. }, JoinableGeo::Poly { edges: b_edges, .. }) => {
            // Pass 1 (A edges vs B): found_touch, boundary crossings, same-side
            // shared edges, and A contained in B (parity).
            let mut acc = PolyTouchesPoly::new(a_edges, b_edges, true);
            a.fold_for_grouped_check(b, &mut acc);
            if acc.disqualified {
                return false;
            }
            let touched = acc.found_touch;

            // Pass 2 (B edges vs A) catches the remaining interior overlap, B
            // contained in A. Crossings/shared edges are symmetric (already seen in
            // pass 1), so this pass only does the containment parity check.
            let mut acc = PolyTouchesPoly::new(b_edges, a_edges, false);
            b.fold_for_grouped_check(a, &mut acc);
            if acc.disqualified {
                return false;
            }
            touched
        }
    }
}

// Whether a hit at parameter t on a segment lies on the linestring's interior.
fn hit_on_interior(seg: &PlanarLineSegment, t: f64) -> bool {
    if t <= EPS {
        !seg.p1_boundary()
    } else if t >= 1.0 - EPS {
        !seg.p2_boundary()
    } else {
        true
    }
}

// *************************************************************
// Accumulators. Each carries found_touch(a qualifying boundary
// contact) and disqualified (a fatal interior–interior contact).
// ready short-circuits only on disqualified.
// *************************************************************

// A point set touches a line if some point lies on a line boundary vertex (odd
// mod-2 endpoint) and no point lies on the line's interior.
pub(crate) struct PointTouchesLine<'a> {
    points: &'a [PlanarPoint],
    lines: &'a [PlanarLineSegment],
    found_touch: bool,
    disqualified: bool,
}

impl<'a> PointTouchesLine<'a> {
    pub(crate) fn new(points: &'a [PlanarPoint], lines: &'a [PlanarLineSegment]) -> Self {
        PointTouchesLine {
            points,
            lines,
            found_touch: false,
            disqualified: false,
        }
    }
}

impl Accumulator for PointTouchesLine<'_> {
    type K = Planar;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        let p = &self.points[li];
        let seg = &self.lines[ri];
        // A coincidence with an endpoint is classified by the mod-2 flag: boundary
        // vertex is a touch, interior vertex disqualifies.
        if let Some(is_boundary) = point_at_line_endpoint(p, seg) {
            if is_boundary {
                self.found_touch = true;
            } else {
                self.disqualified = true;
            }
        } else if let Some(t) = point_on_segment_t::<Planar, _>(p, seg) {
            // Strictly interior to the segment disqualifies, t≈0/1 is an endpoint,
            // already handled above (its status is the mod-2 flag).
            if t > EPS && t < 1.0 - EPS {
                self.disqualified = true;
            }
        }
    }

    fn ready(&self) -> bool {
        self.disqualified
    }

    fn finish(self) -> bool {
        self.found_touch && !self.disqualified
    }
}

// A point set touches a poly if some point is on the boundary (any edge, all of
// which are boundary) and no point is strictly inside.
pub(crate) struct PointTouchesPoly<'a> {
    points: &'a [PlanarPoint],
    edges: &'a [PlanarEdge],
    current_group: Option<usize>,
    parity: bool,
    on_edge: bool,
    found_touch: bool,
    disqualified: bool,
}

impl<'a> PointTouchesPoly<'a> {
    pub(crate) fn new(points: &'a [PlanarPoint], edges: &'a [PlanarEdge]) -> Self {
        PointTouchesPoly {
            points,
            edges,
            current_group: None,
            parity: false,
            on_edge: false,
            found_touch: false,
            disqualified: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.on_edge {
            self.found_touch = true;
        } else if self.parity {
            self.disqualified = true;
        }
    }
}

impl Accumulator for PointTouchesPoly<'_> {
    type K = Planar;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.disqualified {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            self.on_edge = false;
        }

        let p = &self.points[li];
        let edge = &self.edges[ri];
        if point_on_segment_t::<Planar, _>(p, edge).is_some() {
            self.on_edge = true;
        } else if Planar::ray_crosses_edge(p, edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.disqualified
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found_touch && !self.disqualified
    }
}

// Two lines touch iff the share a point that is a boundary of at
// least one side, and no shared point is interior to both. Purely
// pairwise, so dual descent + box_overlap.
pub(crate) struct LineTouchesLine<'a> {
    left: &'a [PlanarLineSegment],
    right: &'a [PlanarLineSegment],
    found_touch: bool,
    disqualified: bool,
}

impl<'a> LineTouchesLine<'a> {
    pub(crate) fn new(left: &'a [PlanarLineSegment], right: &'a [PlanarLineSegment]) -> Self {
        LineTouchesLine {
            left,
            right,
            found_touch: false,
            disqualified: false,
        }
    }

    fn classify(&mut self, a_interior: bool, b_interior: bool) {
        if a_interior && b_interior {
            self.disqualified = true;
        } else {
            self.found_touch = true;
        }
    }
}

impl Accumulator for LineTouchesLine<'_> {
    type K = Planar;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        let a = &self.left[li];
        let b = &self.right[ri];
        if let Some((lo, hi)) = covered_range::<Planar, _, _>(a, b) {
            // Collinear: a positive-length overlap means the interiors overlap.
            if hi - lo > EPS {
                self.disqualified = true;
                return;
            }
            let (t_a, a_pt) = if lo <= EPS {
                (0.0, a.p1())
            } else {
                (1.0, a.p2())
            };
            let a_interior = hit_on_interior(a, t_a);
            let b_interior = point_at_line_endpoint(a_pt, b).is_none_or(|is_boundary| !is_boundary);
            self.classify(a_interior, b_interior);
            return;
        }
        if let Some((t, u)) = segment_intersection::<Planar, _, _>(a, b) {
            self.classify(hit_on_interior(a, t), hit_on_interior(b, u));
        }
    }

    fn ready(&self) -> bool {
        self.disqualified
    }

    fn finish(self) -> bool {
        self.found_touch && !self.disqualified
    }
}

// A line touches a poly if the line's interior never enters the
// poly's interior and there is at least one boundary contact. Grouped
// single descent so ray-parity sees all of a segment's candidate edges
// contiguously,
pub(crate) struct LineTouchesPoly<'a> {
    left: &'a [PlanarLineSegment],
    edges: &'a [PlanarEdge],
    current_group: Option<usize>,
    parity: bool,
    parity_valid: bool,
    found_touch: bool,
    disqualified: bool,
}

impl<'a> LineTouchesPoly<'a> {
    pub(crate) fn new(left: &'a [PlanarLineSegment], edges: &'a [PlanarEdge]) -> Self {
        LineTouchesPoly {
            left,
            edges,
            current_group: None,
            parity: false,
            parity_valid: true,
            found_touch: false,
            disqualified: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.parity_valid && self.parity {
            self.disqualified = true;
        }
    }
}

impl Accumulator for LineTouchesPoly<'_> {
    type K = Planar;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.disqualified {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            self.parity_valid = true;
        }

        let seg = &self.left[li];
        let edge = &self.edges[ri];
        let next = &self.edges[edge.next()];

        if covered_range::<Planar, _, _>(seg, edge).is_some()
            || segments_cross::<Planar, _, _>(seg, edge)
        {
            self.found_touch = true;
        }

        let (qualifying, _) = segment_crossing_check::<Planar, _>(seg, edge, next);
        if qualifying {
            self.disqualified = true;
        }

        let (ray, inside, parity_valid) = midpoint_ray_check::<Planar, _>(seg, edge, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.disqualified = true;
        }
        if !parity_valid {
            self.parity_valid = false;
        }
    }

    fn ready(&self) -> bool {
        self.disqualified
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found_touch && !self.disqualified
    }
}

// A poly touches a poly if their interiors are disjoint and boundaries
// contact. Like LineTouchesPoly but the query is an edge (which has an
// interior side), so it adds the collinear shared-edge case, if their
// interiors are on the sane side, the regions overlap along the edge,
// which disqualifies. Used in both directions: pass 1 does the full check
// plus A-inside-B parity; pass 2 does only the containment parity to
// catch B-inside-A.
pub(crate) struct PolyTouchesPoly<'a> {
    left: &'a [PlanarEdge],
    right: &'a [PlanarEdge],
    check_crossings: bool,
    current_group: Option<usize>,
    parity: bool,
    parity_valid: bool,
    found_touch: bool,
    disqualified: bool,
}

impl<'a> PolyTouchesPoly<'a> {
    pub(crate) fn new(
        left: &'a [PlanarEdge],
        right: &'a [PlanarEdge],
        check_crossings: bool,
    ) -> Self {
        PolyTouchesPoly {
            left,
            right,
            check_crossings,
            current_group: None,
            parity: false,
            parity_valid: true,
            found_touch: false,
            disqualified: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.parity_valid && self.parity {
            self.disqualified = true;
        }
    }
}

impl Accumulator for PolyTouchesPoly<'_> {
    type K = Planar;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.disqualified {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            self.parity_valid = true;
        }

        let a = &self.left[li];
        let b = &self.right[ri];
        let next = &self.right[b.next()];

        if self.check_crossings {
            if let Some((lo, hi)) = covered_range::<Planar, _, _>(a, b) {
                // Collinear: a positive-length overlap is a shared boundary segment,
                // same-side interiors overlap disqualifies, opposite-side is a touch.
                if hi - lo > EPS && interiors_same_side::<Planar>(a, b) {
                    self.disqualified = true;
                } else {
                    self.found_touch = true;
                }
            } else if segments_cross::<Planar, _, _>(a, b) {
                self.found_touch = true;
            }
            // Transversal/vertex crossing, or collinear overlap peeling into the
            // interior → interior entry.
            let (qualifying, _) = segment_crossing_check::<Planar, _>(a, b, next);
            if qualifying {
                self.disqualified = true;
            }
        }

        let (ray, inside, parity_valid) = midpoint_ray_check::<Planar, _>(a, b, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.disqualified = true;
        }
        if !parity_valid {
            self.parity_valid = false;
        }
    }

    fn ready(&self) -> bool {
        self.disqualified
    }

    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.found_touch && !self.disqualified
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    use super::st_touches;
    use crate::geospatial::geos::joinable_geo::{JoinableGeo, Planar};
    pub(super) use crate::geospatial::test_utils::wkt_to_wkb;

    pub(super) fn compare_touches(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.touches(&geos_b).unwrap();

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();

        // Touches is symmetric, so both orderings must match GEOS.
        assert_eq!(geos_result, st_touches(&joinable_a, &joinable_b), "{label}");
        assert_eq!(
            geos_result,
            st_touches(&joinable_b, &joinable_a),
            "{label}, swapped"
        );
    }
}

#[cfg(test)]
mod touches_tests {
    use super::test_helpers::compare_touches;

    #[test]
    fn point_left() {
        compare_touches("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_touches("point left, case 2", "POINT (1 1)", "POINT (2 2)");

        compare_touches("point left, case 3", "POINT (1 0)", "LINESTRING (0 0, 2 0)");
        compare_touches("point left, case 4", "POINT (0 0)", "LINESTRING (0 0, 2 0)");
        compare_touches(
            "point left, case 5",
            "POINT (1 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_touches("point left, case 6", "POINT (5 5)", "LINESTRING (0 0, 2 0)");

        compare_touches(
            "point left, case 7",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_touches(
            "point left, case 8",
            "POINT (5 5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_touches(
            "point left, case 9",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_touches(
            "point left, case 10",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_touches(
            "point left, case 11",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        // Point on a degree-3 junction (an odd mod-2 vertex → line boundary).
        compare_touches(
            "point left, case 12",
            "POINT (2 2)",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4), (2 2, 2 5))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_touches(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2)",
        );
        compare_touches(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (2 2, 3 3)",
        );
        compare_touches(
            "multi point left, case 3",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (5 5, 6 6)",
        );

        compare_touches(
            "multi point left, case 4",
            "MULTIPOINT (0 0, 1 0)",
            "LINESTRING (0 0, 2 0)",
        );
        compare_touches(
            "multi point left, case 5",
            "MULTIPOINT (0 0, 2 0)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_touches(
            "multi point left, case 6",
            "MULTIPOINT (1 0, 4 0)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_touches(
            "multi point left, case 7",
            "MULTIPOINT (1 0, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_touches(
            "line left, case 1",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (2 0, 4 0)",
        );
        compare_touches(
            "line left, case 2",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 -2, 2 2)",
        );
        compare_touches(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 6 0)",
        );
        compare_touches(
            "line left, case 4",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );
        compare_touches(
            "line left, case 5",
            "LINESTRING (0 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
        );
        compare_touches(
            "line left, case 6",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 2 2)",
        );

        compare_touches(
            "line left, case 7",
            "LINESTRING (2 2, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );
        compare_touches(
            "line left, case 8",
            "LINESTRING (-1 3, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );
        compare_touches(
            "line left, case 9",
            "LINESTRING (3 0, 3 -2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );
        compare_touches(
            "line left, case 10",
            "LINESTRING (3 0, 7 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );
        compare_touches(
            "line left, case 11",
            "LINESTRING (2 0, 4 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );
        compare_touches(
            "line left, case 12",
            "LINESTRING (2 2, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))",
        );
        compare_touches(
            "line left, case 13",
            "LINESTRING (2 2, 2 1)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))",
        );

        // Inward notch: the floor edge (3 3)-(5 3) has reflex corners, so a
        // collinear line extending past them enters the interior.
        compare_touches(
            "line left, case 14",
            "LINESTRING (3.5 3, 4.5 3)",
            "POLYGON ((0 0, 3 0, 3 3, 5 3, 5 0, 8 0, 8 8, 0 8, 0 0))",
        );
        compare_touches(
            "line left, case 15",
            "LINESTRING (2 3, 4 3)",
            "POLYGON ((0 0, 3 0, 3 3, 5 3, 5 0, 8 0, 8 8, 0 8, 0 0))",
        );
        compare_touches(
            "line left, case 16",
            "LINESTRING (4 3, 6 3)",
            "POLYGON ((0 0, 3 0, 3 3, 5 3, 5 0, 8 0, 8 8, 0 8, 0 0))",
        );

        // Outward tab: the top edge (5 11)-(3 11) has convex corners, so a
        // collinear line extending past them stays in the exterior.
        compare_touches(
            "line left, case 17",
            "LINESTRING (3.5 11, 4.5 11)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 11, 3 11, 3 8, 0 8, 0 0))",
        );
        compare_touches(
            "line left, case 18",
            "LINESTRING (2 11, 4 11)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 11, 3 11, 3 8, 0 8, 0 0))",
        );
        compare_touches(
            "line left, case 19",
            "LINESTRING (4 11, 6 11)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 11, 3 11, 3 8, 0 8, 0 0))",
        );
    }

    #[test]
    fn multi_line_left() {
        // Both junctions degree-2 (interior vertices), meeting there → II.
        compare_touches(
            "multi line left, case 1",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4))",
        );
        // Right junction degree-3 (boundary), left degree-2 (interior) → IB touch.
        compare_touches(
            "multi line left, case 2",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4), (2 2, 2 5))",
        );
        // Both junctions degree-3 (boundary) → BB touch.
        compare_touches(
            "multi line left, case 3",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0), (2 2, 0 3))",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4), (2 2, 2 5))",
        );
        // One segment on a poly edge, the other inside the other poly → II.
        compare_touches(
            "multi line left, case 4",
            "MULTILINESTRING ((1 0, 3 0), (7 1, 8 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );
    }

    #[test]
    fn poly_left() {
        compare_touches(
            "poly left, case 1",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_touches(
            "poly left, case 2",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_touches(
            "poly left, case 3",
            "POLYGON ((1 -2, 3 -2, 3 0, 1 0, 1 -2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_touches(
            "poly left, case 4",
            "POLYGON ((1 0, 3 0, 3 2, 1 2, 1 0))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_touches(
            "poly left, case 5",
            "POLYGON ((2.5 2, 3.5 2, 3.5 3, 2.5 3, 2.5 2))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_touches(
            "poly left, case 6",
            "POLYGON ((2.5 1, 3.5 1, 3.5 2, 2.5 2, 2.5 1))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_touches(
            "poly left, case 7",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((5 5, 7 5, 7 7, 5 7, 5 5))",
        );
        compare_touches(
            "poly left, case 8",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_touches(
            "poly left, case 9",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
        );
        compare_touches(
            "poly left, case 10",
            "POLYGON ((2 -2, 4 -2, 3 0, 2 -2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        // Inward notch: left inside right, its edge fully contained in the notch
        // floor, interiors same side.
        compare_touches(
            "poly left, case 11",
            "POLYGON ((3.5 3, 4.5 3, 4.5 4, 3.5 4, 3.5 3))",
            "POLYGON ((0 0, 3 0, 3 3, 5 3, 5 0, 8 0, 8 8, 0 8, 0 0))",
        );
        // Outward tab: left above the tab sharing its top edge, interiors opposite.
        // Tab edge fully contains left's edge.
        compare_touches(
            "poly left, case 12",
            "POLYGON ((3.5 11, 4.5 11, 4.5 12, 3.5 12, 3.5 11))",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 11, 3 11, 3 8, 0 8, 0 0))",
        );
        // Left's edge fully contains the tab's top edge.
        compare_touches(
            "poly left, case 13",
            "POLYGON ((2 11, 6 11, 6 12, 2 12, 2 11))",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 11, 3 11, 3 8, 0 8, 0 0))",
        );
    }

    #[test]
    fn multi_poly_left() {
        // One pair touches on a shared edge, but another pair's interiors overlap → II.
        compare_touches(
            "multi poly left, case 1",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 13 10, 13 13, 10 13, 10 10)))",
            "MULTIPOLYGON (((2 0, 4 0, 4 2, 2 2, 2 0)), ((11 11, 14 11, 14 14, 11 14, 11 11)))",
        );
        // One pair touches on a shared edge, the rest disjoint → touch.
        compare_touches(
            "multi poly left, case 2",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((2 0, 4 0, 4 2, 2 2, 2 0)), ((20 20, 22 20, 22 22, 20 22, 20 20)))",
        );
        // No contact at all.
        compare_touches(
            "multi poly left, case 3",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((30 30, 32 30, 32 32, 30 32, 30 30)), ((40 40, 42 40, 42 42, 40 42, 40 40)))",
        );
    }
}
