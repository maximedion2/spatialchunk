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
    covered_range, point_on_segment_t, segments_cross, Accumulator, Bboxable, Edge, GeoKernel,
    JoinableGeo, Point, RefStatus, Segment,
};

// *************************************************************
// st_intersects(a, b) -> true if a and b share at least one point. Unlike
// st_within, this is existential (the first genuine touch wins) and symmetric.
// *************************************************************
pub(crate) fn st_intersects<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> bool {
    match (a, b) {
        (JoinableGeo::Point { points: a_pts, .. }, JoinableGeo::Point { points: b_pts, .. }) => {
            let mut acc = PointIntersectsPoint::<K>::new(a_pts, b_pts);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points, .. }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointIntersectsLine::<K>::new(points, lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }
        (
            JoinableGeo::Point { points, .. },
            JoinableGeo::Poly {
                edges, ref_status, ..
            },
        ) => {
            let mut acc = PointIntersectsPoly::<K>::new(points, edges, ref_status);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            let mut acc = LineIntersectsLine::<K>::new(l_lines, r_lines);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (
            JoinableGeo::Line { lines, .. },
            JoinableGeo::Poly {
                edges, ref_status, ..
            },
        ) => {
            let mut acc = SegmentsIntersectPoly::<K, _>::new(lines, edges, ref_status, true);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (
            JoinableGeo::Poly {
                edges: a_edges,
                ref_status: a_ref,
                ..
            },
            JoinableGeo::Poly {
                edges: b_edges,
                ref_status: b_ref,
                ..
            },
        ) => {
            // First pass (A against B): boundary crossings and A contained in B.
            let mut acc = SegmentsIntersectPoly::<K, _>::new(a_edges, b_edges, b_ref, true);
            a.fold_for_grouped_check(b, &mut acc);
            if acc.finish() {
                return true;
            }
            // Second pass (B against A) catches the remaining case, B contained in
            // A. No crossings exist by now, so skip the boundary-contact test.
            let mut acc = SegmentsIntersectPoly::<K, _>::new(b_edges, a_edges, a_ref, false);
            b.fold_for_grouped_check(a, &mut acc);
            acc.finish()
        }

        // Symmetric: forward to the point-on-left orientation.
        (JoinableGeo::Line { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => st_intersects(b, a),
    }
}

// *************************************************************
// Accumulators. Each carries a `found` flag flipped on the first genuine
// geometric touch; `ready` and `finish` both just report it.
// *************************************************************

// Two point sets intersect if any left point equals any right point. Exact
// equality, matching PointInPoint in st_within.
pub(crate) struct PointIntersectsPoint<'a, K: GeoKernel> {
    left: &'a [K::Point],
    right: &'a [K::Point],
    found: bool,
}

impl<'a, K: GeoKernel> PointIntersectsPoint<'a, K> {
    pub(crate) fn new(left: &'a [K::Point], right: &'a [K::Point]) -> Self {
        PointIntersectsPoint {
            left,
            right,
            found: false,
        }
    }
}

impl<K: GeoKernel> Accumulator for PointIntersectsPoint<'_, K> {
    type K = K;

    // No ray casting, so a symmetric bbox overlap is the correct, tighter prune.
    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if self.left[li].is_equal(&self.right[ri]) {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A point set and a line intersect if any point lies on any segment (interior
// or endpoint — intersects doesn't distinguish). No ray casting, so box_overlap.
pub(crate) struct PointIntersectsLine<'a, K: GeoKernel> {
    points: &'a [K::Point],
    lines: &'a [K::Line],
    found: bool,
}

impl<'a, K: GeoKernel> PointIntersectsLine<'a, K> {
    pub(crate) fn new(points: &'a [K::Point], lines: &'a [K::Line]) -> Self {
        PointIntersectsLine {
            points,
            lines,
            found: false,
        }
    }
}

impl<K: GeoKernel> Accumulator for PointIntersectsLine<'_, K> {
    type K = K;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if point_on_segment_t::<K, _>(&self.points[li], &self.lines[ri]).is_some() {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A point set intersects a poly if any point is on the boundary or inside.
// Grouped single descent (point as query) so ray-parity sees all of a point's
// candidate edges contiguously; default ray_candidate prune. A point on an edge
// is an immediate hit; otherwise it's inside when its ray-crossing parity differs
// from the ray end's status.
//
// A point the descent matches against no edge has no crossings, so it's inside
// iff its ray's end is. Those are tracked (seen) only when that can happen.
pub(crate) struct PointIntersectsPoly<'a, K: GeoKernel> {
    points: &'a [K::Point],
    edges: &'a [K::Edge],
    ref_status: &'a RefStatus,
    seen: Option<Vec<bool>>,
    current_group: Option<usize>,
    parity: bool,
    found: bool,
}

impl<'a, K: GeoKernel> PointIntersectsPoly<'a, K> {
    pub(crate) fn new(
        points: &'a [K::Point],
        edges: &'a [K::Edge],
        ref_status: &'a RefStatus,
    ) -> Self {
        PointIntersectsPoly {
            points,
            edges,
            ref_status,
            seen: (!ref_status.always_outside()).then(|| vec![false; points.len()]),
            current_group: None,
            parity: false,
            found: false,
        }
    }

    fn ray_end_inside(&self, li: usize) -> bool {
        !self.ref_status.always_outside()
            && self
                .ref_status
                .inside_for::<K>(self.edges, &self.points[li])
    }

    fn finalize_group(&mut self, li: usize) {
        if self.parity != self.ray_end_inside(li) {
            self.found = true;
        }
    }
}

impl<K: GeoKernel> Accumulator for PointIntersectsPoly<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
                if self.found {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            if let Some(seen) = &mut self.seen {
                seen[li] = true;
            }
        }

        let p = &self.points[li];
        let edge = &self.edges[ri];
        if point_on_segment_t::<K, _>(p, edge).is_some() {
            self.found = true;
        } else if K::ray_crosses_edge(p, edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(mut self) -> bool {
        if let Some(li) = self.current_group {
            self.finalize_group(li);
        }
        self.found
            || self.seen.as_ref().is_some_and(|seen| {
                (0..self.points.len()).any(|li| !seen[li] && self.ray_end_inside(li))
            })
    }
}

// Two line sets intersect if any left segment touches any right segment: either a
// non-parallel crossing (`segments_cross`) or a collinear overlap (`covered_range`).
// Purely pairwise, so dual descent + box_overlap.
pub(crate) struct LineIntersectsLine<'a, K: GeoKernel> {
    left: &'a [K::Line],
    right: &'a [K::Line],
    found: bool,
}

impl<'a, K: GeoKernel> LineIntersectsLine<'a, K> {
    pub(crate) fn new(left: &'a [K::Line], right: &'a [K::Line]) -> Self {
        LineIntersectsLine {
            left,
            right,
            found: false,
        }
    }
}

impl<K: GeoKernel> Accumulator for LineIntersectsLine<'_, K> {
    type K = K;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        let (l, r) = (&self.left[li], &self.right[ri]);
        if covered_range::<K, _, _>(l, r).is_some() || segments_cross::<K, _, _>(l, r) {
            self.found = true;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(self) -> bool {
        self.found
    }
}

// A set of query segments intersects a poly if any segment touches the boundary,
// or any segment lies inside (fully contained, boundaries never crossed). Grouped
// single descent so ray-parity sees all of a segment's candidate edges
// contiguously. Boundary contact uses the inclusive segments_cross/covered_range.
// Generic over the query segment type so it serves both line×poly (query =
// K::Line) and poly×poly (query = K::Edge).
//
// As for PointIntersectsPoly, a segment matched against no edge is inside iff the
// end of the ray cast from its midpoint is.
//
// `check_crossings` skips the boundary-contact test when the caller already knows
// there are none. the second poly×poly pass (B against A) runs only after the
// first pass found no contact, so it only needs the containment parity check.
pub(crate) struct SegmentsIntersectPoly<'a, K: GeoKernel, L: Segment<P = K::Point>> {
    left: &'a [L],
    edges: &'a [K::Edge],
    ref_status: &'a RefStatus,
    check_crossings: bool,
    seen: Option<Vec<bool>>,
    current_group: Option<usize>,
    parity: bool,
    found: bool,
}

impl<'a, K: GeoKernel, L: Segment<P = K::Point>> SegmentsIntersectPoly<'a, K, L> {
    pub(crate) fn new(
        left: &'a [L],
        edges: &'a [K::Edge],
        ref_status: &'a RefStatus,
        check_crossings: bool,
    ) -> Self {
        SegmentsIntersectPoly {
            left,
            edges,
            ref_status,
            check_crossings,
            seen: (!ref_status.always_outside()).then(|| vec![false; left.len()]),
            current_group: None,
            parity: false,
            found: false,
        }
    }

    fn ray_end_inside(&self, li: usize) -> bool {
        !self.ref_status.always_outside()
            && self
                .ref_status
                .inside_for::<K>(self.edges, &K::midpoint(&self.left[li]))
    }

    fn finalize_group(&mut self, li: usize) {
        if self.parity != self.ray_end_inside(li) {
            self.found = true;
        }
    }
}

impl<K: GeoKernel, L: Segment<P = K::Point>> Accumulator for SegmentsIntersectPoly<'_, K, L> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
                if self.found {
                    return;
                }
            }
            self.current_group = Some(li);
            self.parity = false;
            if let Some(seen) = &mut self.seen {
                seen[li] = true;
            }
        }

        let seg = &self.left[li];
        let edge = &self.edges[ri];
        // Boundary contact (inclusive of endpoints/vertices) is an immediate hit.
        if self.check_crossings
            && (covered_range::<K, _, _>(seg, edge).is_some()
                || segments_cross::<K, _, _>(seg, edge))
        {
            self.found = true;
            return;
        }
        // Otherwise cast a ray from the segment midpoint for the fully-inside case.
        // The midpoint can't be on the boundary here, that would be a point shared
        // with this same edge, already caught above (or, when crossings aren't
        // checked, ruled out by the prior pass).
        if K::ray_crosses_edge(&K::midpoint(seg), edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.found
    }

    fn finish(mut self) -> bool {
        if let Some(li) = self.current_group {
            self.finalize_group(li);
        }
        self.found
            || self.seen.as_ref().is_some_and(|seen| {
                (0..self.left.len()).any(|li| !seen[li] && self.ray_end_inside(li))
            })
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    use super::st_intersects;
    use crate::geospatial::geos::joinable_geo::{JoinableGeo, Planar, Spherical};
    pub(super) use crate::geospatial::test_utils::wkt_to_wkb;

    pub(super) fn compare_intersects(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.intersects(&geos_b).unwrap();

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();

        // Intersects is symmetric, so both orderings must match GEOS.
        assert_eq!(
            geos_result,
            st_intersects(&joinable_a, &joinable_b),
            "{label}"
        );
        assert_eq!(
            geos_result,
            st_intersects(&joinable_b, &joinable_a),
            "{label}, swapped"
        );
    }

    // No oracle for geographies, so the expected result is given.
    pub(super) fn check_geography_intersects(
        label: &str,
        wkt_a: &str,
        wkt_b: &str,
        expected: bool,
    ) {
        let joinable_a = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_a)).unwrap();
        let joinable_b = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_b)).unwrap();

        assert_eq!(expected, st_intersects(&joinable_a, &joinable_b), "{label}");
        assert_eq!(
            expected,
            st_intersects(&joinable_b, &joinable_a),
            "{label}, swapped"
        );
    }
}

#[cfg(test)]
mod intersects_tests {
    use super::test_helpers::compare_intersects;

    #[test]
    fn point_left() {
        compare_intersects("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_intersects("point left, case 2", "POINT (1 1)", "POINT (2 2)");
        compare_intersects("point left, case 3", "POINT (1 1)", "MULTIPOINT (0 0, 1 1)");
        compare_intersects("point left, case 4", "POINT (1 1)", "MULTIPOINT (0 0, 2 2)");

        compare_intersects(
            "point left, case 5",
            "POINT (0.5 0)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects("point left, case 6", "POINT (2 0)", "LINESTRING (0 0, 1 0)");
        compare_intersects("point left, case 7", "POINT (0 0)", "LINESTRING (0 0, 1 0)");
        compare_intersects(
            "point left, case 8",
            "POINT (0.5 0)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "point left, case 9",
            "POINT (5 5)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "point left, case 10",
            "POINT (1 0)",
            "MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))",
        );

        compare_intersects(
            "point left, case 11",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 12",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "point left, case 13",
            "POINT (5 5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 14",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "point left, case 15",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 16",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "point left, case 17",
            "POINT (2 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_intersects(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (1 1)",
        );
        compare_intersects(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (3 3)",
        );

        compare_intersects(
            "multi point left, case 3",
            "MULTIPOINT (0.5 0, 5 5)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects(
            "multi point left, case 4",
            "MULTIPOINT (5 5, 6 6)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_intersects(
            "multi point left, case 5",
            "MULTIPOINT (1 0, 5 5)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_intersects(
            "multi point left, case 6",
            "MULTIPOINT (5 5, 6 6)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );
        compare_intersects(
            "multi point left, case 7",
            "MULTIPOINT (0.5 0, 0.5 1)",
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        );

        compare_intersects(
            "multi point left, case 8",
            "MULTIPOINT (1 1, 5 5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "multi point left, case 9",
            "MULTIPOINT (5 5, 6 6)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "multi point left, case 10",
            "MULTIPOINT (2 2, 2.5 2.5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "multi point left, case 11",
            "MULTIPOINT (1 1, 8 8)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_intersects("line left, case 1", "LINESTRING (0 0, 4 0)", "POINT (2 0)");
        compare_intersects("line left, case 2", "LINESTRING (0 0, 4 0)", "POINT (2 2)");

        compare_intersects(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 -2, 2 2)",
        );
        compare_intersects(
            "line left, case 4",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 6 0)",
        );
        compare_intersects(
            "line left, case 5",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );
        compare_intersects(
            "line left, case 6",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (1 0, 3 0)",
        );
        compare_intersects(
            "line left, case 7",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 0, 2 2)",
        );
        compare_intersects(
            "line left, case 8",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (4 0, 6 2)",
        );
        compare_intersects(
            "line left, case 9",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 2, 4 2)",
        );
        compare_intersects(
            "line left, case 10",
            "LINESTRING (0 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
        );

        compare_intersects(
            "line left, case 11",
            "LINESTRING (-1 1, 3 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 12",
            "LINESTRING (0.5 0.5, 1.5 1.5)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 13",
            "LINESTRING (0.5 0, 1.5 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 14",
            "LINESTRING (5 5, 6 6)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );
        compare_intersects(
            "line left, case 15",
            "LINESTRING (1.5 2, 2.5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "line left, case 16",
            "LINESTRING (2 2, 2 0.5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "line left, case 17",
            "LINESTRING (-1 1, 1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "line left, case 18",
            "LINESTRING (10 10, 11 11)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn multi_line_left() {
        compare_intersects(
            "multi line left, case 1",
            "MULTILINESTRING ((0 0, 2 2), (2 2, 4 0))",
            "MULTILINESTRING ((0 4, 2 2), (2 2, 4 4))",
        );
        compare_intersects(
            "multi line left, case 2",
            "MULTILINESTRING ((2 2, 4 2), (4 2, 6 2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_intersects(
            "multi line left, case 3",
            "MULTILINESTRING ((3 3, 3 2), (3 2, 3 1))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_intersects(
            "multi line left, case 4",
            "MULTILINESTRING ((-1 1, 1 1), (10 10, 11 11))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
        compare_intersects(
            "multi line left, case 5",
            "MULTILINESTRING ((10 10, 11 11), (12 12, 13 13))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );
    }

    #[test]
    fn poly_left() {
        compare_intersects(
            "poly left, case 1",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        );
        compare_intersects(
            "poly left, case 2",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((5 5, 7 5, 7 7, 5 7, 5 5))",
        );
        compare_intersects(
            "poly left, case 3",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_intersects(
            "poly left, case 4",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
        );
        compare_intersects(
            "poly left, case 5",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 5 1, 5 5, 1 5, 1 1))",
        );
        compare_intersects(
            "poly left, case 6",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 5 1, 5 5, 1 5, 1 1))",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
        );
        compare_intersects(
            "poly left, case 7",
            "POLYGON ((1 1, 4 1, 4 4, 1 4, 1 1))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
        compare_intersects(
            "poly left, case 8",
            "POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 0, 7 0, 7 2, 5 2, 5 0)))",
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_intersects(
            "multi poly left, case 1",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((1 1, 3 1, 3 3, 1 3, 1 1)), ((20 20, 22 20, 22 22, 20 22, 20 20)))",
        );
        compare_intersects(
            "multi poly left, case 2",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            "MULTIPOLYGON (((30 30, 32 30, 32 32, 30 32, 30 30)), ((40 40, 42 40, 42 42, 40 42, 40 40)))",
        );
        compare_intersects(
            "multi poly left, case 3",
            "MULTIPOLYGON (((4 4, 6 4, 6 6, 4 6, 4 4)), ((30 30, 32 30, 32 32, 30 32, 30 30)))",
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2)), ((20 0, 22 0, 22 2, 20 2, 20 0)))",
        );
        compare_intersects(
            "multi poly left, case 4",
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2)), ((30 30, 32 30, 32 32, 30 32, 30 30)))",
            "MULTIPOLYGON (((4 4, 6 4, 6 6, 4 6, 4 4)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
        );
    }

    #[test]
    fn empty_geo() {
        let sq = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        compare_intersects("empty multipoint left", "MULTIPOINT EMPTY", sq);
        compare_intersects("empty multilinestring left", "MULTILINESTRING EMPTY", sq);
        compare_intersects("empty polygon left", "POLYGON EMPTY", sq);
        compare_intersects("empty multipolygon left", "MULTIPOLYGON EMPTY", sq);
    }
}

#[cfg(test)]
mod geography_intersects_tests {
    use super::test_helpers::check_geography_intersects;

    // Great-circle edges between two points at the same latitude bulge towards
    // the pole: from (0 60) to (90 60), the edge peaks at latitude 67.79 at
    // longitude 45, and from (0 50) to (90 50) at 59.32.
    #[test]
    fn edge_bulge() {
        check_geography_intersects(
            "edge bulge, line crosses the bulge",
            "LINESTRING (0 60, 90 60)",
            "LINESTRING (45 62, 45 70)",
            true,
        );
        check_geography_intersects(
            "edge bulge, point inside the bulged poly",
            "POINT (45 65)",
            "POLYGON ((0 50, 90 50, 90 60, 0 60, 0 50))",
            true,
        );
        check_geography_intersects(
            "edge bulge, point under the bulged bottom edge",
            "POINT (45 55)",
            "POLYGON ((0 50, 90 50, 90 60, 0 60, 0 50))",
            false,
        );
    }

    // A cap around the north pole, its ring at latitude 70 going east (the pole
    // on its left), its edges peaking at 75.57 at longitude 45. The hole's ring at
    // latitude 80 goes west, so the pole is in the hole, its edges peaking at
    // 82.89. Neither ring touches the pole.
    #[test]
    fn north_pole_inside() {
        let cap = "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70))";
        let cap_with_hole = "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70), \
            (0 80, -90 80, 180 80, 90 80, 0 80))";

        check_geography_intersects("north pole cap, point inside", "POINT (45 80)", cap, true);
        check_geography_intersects("north pole cap, point outside", "POINT (45 60)", cap, false);
        check_geography_intersects(
            "north pole cap, point at the pole",
            "POINT (0 90)",
            cap,
            true,
        );
        check_geography_intersects(
            "north pole cap with hole, point inside",
            "POINT (45 78)",
            cap_with_hole,
            true,
        );
        check_geography_intersects(
            "north pole cap with hole, point outside",
            "POINT (45 60)",
            cap_with_hole,
            false,
        );
        check_geography_intersects(
            "north pole cap with hole, point in the hole",
            "POINT (45 88)",
            cap_with_hole,
            false,
        );
    }

    // Polys whose shell goes through the north pole, so whether the end of a ray
    // (just short of the pole) is inside depends on the ray's longitude. The outside
    // points are where the rays reach the pole outside the poly without crossing
    // anything.
    // - vertex_cap: a wedge with a vertex at the pole, between the meridians 0 and
    //   90 above its bottom edge (peaking at 75.57 at longitude 45).
    // - edge_cap: a triangle whose edge from (135 80) to (-45 80) goes over the
    //   pole, along the meridians 135 and -45, the triangle being on the longitude
    //   45 side of it.
    // - square_hole: between the meridians 40 and 50, its edges peaking at 77.05
    //   and 80.04, touching neither the pole nor the shells.
    // - pole_hole: a narrower wedge with a vertex at the pole, between the meridians
    //   20 and 70 above its bottom edge (peaking at 80.92).
    #[test]
    fn north_pole_touching() {
        let vertex_cap = "POLYGON ((0 70, 90 70, 0 90, 0 70))";
        let edge_cap = "POLYGON ((-45 80, 45 70, 135 80, -45 80))";
        let square_hole = "(40 77, 40 80, 50 80, 50 77, 40 77)";
        let pole_hole = "(20 80, 0 90, 70 80, 20 80)";
        let vertex_cap_square_hole = format!("POLYGON ((0 70, 90 70, 0 90, 0 70), {square_hole})");
        let edge_cap_square_hole =
            format!("POLYGON ((-45 80, 45 70, 135 80, -45 80), {square_hole})");
        let vertex_cap_pole_hole = format!("POLYGON ((0 70, 90 70, 0 90, 0 70), {pole_hole})");

        check_geography_intersects(
            "vertex cap, point inside",
            "POINT (45 85)",
            vertex_cap,
            true,
        );
        check_geography_intersects(
            "vertex cap, point outside",
            "POINT (135 80)",
            vertex_cap,
            false,
        );
        check_geography_intersects(
            "vertex cap with square hole, point inside",
            "POINT (45 85)",
            &vertex_cap_square_hole,
            true,
        );
        check_geography_intersects(
            "vertex cap with square hole, point outside",
            "POINT (135 80)",
            &vertex_cap_square_hole,
            false,
        );
        check_geography_intersects(
            "vertex cap with square hole, point in the hole",
            "POINT (45 78.5)",
            &vertex_cap_square_hole,
            false,
        );

        check_geography_intersects("edge cap, point inside", "POINT (45 85)", edge_cap, true);
        check_geography_intersects(
            "edge cap, point outside",
            "POINT (-135 80)",
            edge_cap,
            false,
        );
        check_geography_intersects(
            "edge cap with square hole, point inside",
            "POINT (45 85)",
            &edge_cap_square_hole,
            true,
        );
        check_geography_intersects(
            "edge cap with square hole, point outside",
            "POINT (-135 80)",
            &edge_cap_square_hole,
            false,
        );
        check_geography_intersects(
            "edge cap with square hole, point in the hole",
            "POINT (45 78.5)",
            &edge_cap_square_hole,
            false,
        );

        check_geography_intersects(
            "vertex cap with pole hole, point inside",
            "POINT (45 78)",
            &vertex_cap_pole_hole,
            true,
        );
        check_geography_intersects(
            "vertex cap with pole hole, point outside",
            "POINT (135 80)",
            &vertex_cap_pole_hole,
            false,
        );
        check_geography_intersects(
            "vertex cap with pole hole, point in the hole",
            "POINT (45 85)",
            &vertex_cap_pole_hole,
            false,
        );
    }

    // - south_cap: a cap around the south pole, its ring at latitude -70 going west
    //   (the pole on its left), its edges bulging to -75.57 at longitude 45.
    // - both_poles: a square ring going clockwise, so the poly is everything but
    //   the square, both poles included.
    #[test]
    fn south_pole_inside() {
        let south_cap = "POLYGON ((0 -70, -90 -70, 180 -70, 90 -70, 0 -70))";
        let both_poles = "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))";

        check_geography_intersects(
            "south pole cap, point inside",
            "POINT (45 -80)",
            south_cap,
            true,
        );
        check_geography_intersects(
            "south pole cap, point outside",
            "POINT (45 -60)",
            south_cap,
            false,
        );
        check_geography_intersects(
            "both poles, point inside",
            "POINT (45 -60)",
            both_poles,
            true,
        );
        check_geography_intersects(
            "both poles, point outside (in the square)",
            "POINT (5 5)",
            both_poles,
            false,
        );
    }

    // Same edges, one degree short of the bulges.
    #[test]
    fn edge_bulge_near_miss() {
        check_geography_intersects(
            "edge bulge near miss, line stops under the bulge",
            "LINESTRING (0 60, 90 60)",
            "LINESTRING (45 62, 45 66.79)",
            false,
        );
        check_geography_intersects(
            "edge bulge near miss, point above the bulged top edge",
            "POINT (45 68.79)",
            "POLYGON ((0 50, 90 50, 90 60, 0 60, 0 50))",
            false,
        );
        check_geography_intersects(
            "edge bulge near miss, point just under the bulged bottom edge",
            "POINT (45 58.32)",
            "POLYGON ((0 50, 90 50, 90 60, 0 60, 0 50))",
            false,
        );
    }
}
