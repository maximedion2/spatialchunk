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

use std::ops::Range;

use crate::geospatial::geos::joinable_geo::{
    covered_range, midpoint_ray_check, point_at_line_endpoint, point_on_segment_t,
    poly_edge_relation, segment_crossing_check, Accumulator, Bboxable, Edge, GeoKernel,
    JoinableGeo, Point, RefStatus, EPS,
};

// *************************************************************
// The main function that checks for the within condition, within(a, b) -> true if
// a is contained in b.
// *************************************************************
pub(crate) fn st_within<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> bool {
    within_impl(a, b, false)
}

// st_contains is fowarded to st_within with swapped arguments.
pub(crate) fn st_contains<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> bool {
    st_within(b, a)
}

// Shared by st_within and st_covered_by, which only differ by whether a left
// component lying on the right side's boundary qualifies. That only changes the
// outcome when dim(a) < dim(b) (point×line, point×poly, line×poly).
pub(crate) fn within_impl<K: GeoKernel>(
    a: &JoinableGeo<K>,
    b: &JoinableGeo<K>,
    right_side_boundary_qualifies: bool,
) -> bool {
    match (a, b) {
        // A higher-dimensional geometry can't be within a lower-dimensional one.
        (JoinableGeo::Line { .. } | JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => false,

        (
            JoinableGeo::Point { points, .. },
            JoinableGeo::Poly {
                edges, ref_status, ..
            },
        ) => {
            let mut acc =
                PointInPoly::<K>::new(points, edges, ref_status, right_side_boundary_qualifies);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points: a_pts, .. }, JoinableGeo::Point { points: b_pts, .. }) => {
            let mut acc = PointInPoint::<K>::new(a_pts, b_pts);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points, .. }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointInLine::<K>::new(points, lines, right_side_boundary_qualifies);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            let mut acc = LineInLine::<K>::new(l_lines, r_lines);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        (
            JoinableGeo::Line { lines, .. },
            JoinableGeo::Poly {
                edges, ref_status, ..
            },
        ) => {
            let mut acc =
                LineInPoly::<K>::new(lines, edges, ref_status, right_side_boundary_qualifies);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }

        // Polygon within polygon. First a reverse pre-check: if any right ring that can
        // bound a part of the right side's complement lies entirely inside the left poly,
        // then left is not within right. Detecting this requires checking for the right
        // ring within the left poly, so it must be handled separately.
        (
            JoinableGeo::Poly {
                edges: l_edges,
                poly_ids: l_ids,
                ref_status: l_ref,
                ..
            },
            JoinableGeo::Poly {
                edges: r_edges,
                poly_ids: r_ids,
                ref_status: r_ref,
                ..
            },
        ) => {
            let mut ring_acc = ComplementRingsInLeftPoly::<K>::new(
                r_edges,
                b.complement_rings_start(),
                l_edges,
                l_ref,
            );
            b.fold_complement_rings_into(a, &mut ring_acc);
            if ring_acc.finish() {
                return false; // this is the case where left contains a right ring.
            }
            let mut acc = PolyInPoly::<K>::new(l_edges, l_ids, r_edges, r_ids, r_ref);
            a.fold_for_grouped_check(b, &mut acc);
            acc.finish()
        }
    }
}

// *************************************************************
// The accumulators for each combination of left and right geometry The general logic
// is to loop over all the left side components (st_within is a "grouped" check) and
// when one component is done, we check if it is contained by anything on the right.
// Some combinations allow for left to be on right's boundary, this is neither
// qualifying nor disqualifying, so as long as there is one left component that
// touches the right side's interior, all other components can be on a boundary,
// that's fine.
//
// Left side components that we never see, because they didn't match anything on the
// right during index traversal, are an automatic false, unless the right side is a
// poly. A component matched against no edge has no crossings, so it's inside iff the
// end of its ray is, which can't be the case on the plane. Left ids arrive in order,
// so those are the gaps between the groups.
// *************************************************************

// Accumulator for point within poly. Each point is classified inside,
// on boundary or outside.
pub(crate) struct PointInPoly<'a, K: GeoKernel> {
    points: &'a [K::Point],
    edges: &'a [K::Edge],
    ref_status: &'a RefStatus,
    current_group: Option<usize>,
    next_unseen: usize,
    parity: bool,
    on_edge: bool,
    pt_inside: bool,
    pt_outside: bool,
    right_side_boundary_qualifies: bool,
}

impl<'a, K: GeoKernel> PointInPoly<'a, K> {
    pub(crate) fn new(
        points: &'a [K::Point],
        edges: &'a [K::Edge],
        ref_status: &'a RefStatus,
        right_side_boundary_qualifies: bool,
    ) -> Self {
        PointInPoly {
            points,
            edges,
            ref_status,
            current_group: None,
            next_unseen: 0,
            parity: false,
            on_edge: false,
            pt_inside: false,
            pt_outside: false,
            right_side_boundary_qualifies,
        }
    }

    fn ray_end_inside(&self, li: usize) -> bool {
        !self.ref_status.always_outside()
            && self
                .ref_status
                .inside_for::<K>(self.edges, &self.points[li])
    }

    fn finalize_group(&mut self, li: usize) {
        if self.on_edge {
            // On the poly boundary overrides interior: neither for Within,
            // qualifying only under CoveredBy.
            if self.right_side_boundary_qualifies {
                self.pt_inside = true;
            }
        } else if self.parity != self.ray_end_inside(li) {
            self.pt_inside = true;
        } else {
            self.pt_outside = true;
        }
    }

    // Points that matched no edge, never on the boundary.
    fn classify_unseen(&mut self, lis: Range<usize>) {
        if lis.is_empty() {
            return;
        }
        if self.ref_status.always_outside() {
            self.pt_outside = true;
            return;
        }
        for li in lis {
            if self.ray_end_inside(li) {
                self.pt_inside = true;
            } else {
                self.pt_outside = true;
                return;
            }
        }
    }
}

impl<K: GeoKernel> Accumulator for PointInPoly<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
                if self.pt_outside {
                    return;
                }
            }
            self.classify_unseen(self.next_unseen..li);
            if self.pt_outside {
                return;
            }

            self.next_unseen = li + 1;
            self.current_group = Some(li);
            self.parity = false;
            self.on_edge = false;
        }

        let p = &self.points[li];
        let edge = &self.edges[ri];
        if point_on_segment_t::<K, _>(p, edge).is_some() {
            self.on_edge = true;
        } else if K::ray_crosses_edge(p, edge, &self.edges[edge.next()]) {
            self.parity = !self.parity;
        }
    }

    fn ready(&self) -> bool {
        self.pt_outside
    }

    // Within if no point outside, and at least one strictly inside.
    fn finish(mut self) -> bool {
        if let Some(li) = self.current_group {
            if !self.pt_outside {
                self.finalize_group(li);
            }
        }
        if !self.pt_outside {
            self.classify_unseen(self.next_unseen..self.points.len());
        }
        !self.pt_outside && self.pt_inside
    }
}

// Accumulator for point within point. No boundary here, a point is inside
// or outside.
pub(crate) struct PointInPoint<'a, K: GeoKernel> {
    left: &'a [K::Point],
    right: &'a [K::Point],
    current_group: Option<usize>,
    point_matched: bool,
    point_unmatched: bool,
    count: usize,
}

impl<'a, K: GeoKernel> PointInPoint<'a, K> {
    pub(crate) fn new(left: &'a [K::Point], right: &'a [K::Point]) -> Self {
        PointInPoint {
            count: left.len(),
            left,
            right,
            current_group: None,
            point_matched: false,
            point_unmatched: false,
        }
    }
}

impl<K: GeoKernel> Accumulator for PointInPoint<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group == Some(li) {
            if self.point_matched {
                return;
            }
        } else {
            if self.current_group.is_some() && !self.point_matched {
                self.point_unmatched = true; // previous group ended unmatched
            }
            self.current_group = Some(li);
            self.point_matched = false;
        }

        if self.left[li].is_equal(&self.right[ri]) {
            self.point_matched = true;
            self.count -= 1;
        }
    }

    fn ready(&self) -> bool {
        self.count == 0 || self.point_unmatched
    }

    // Within if every left point matched. `point_matched` only guards
    // the empty (n == 0) case; `count == 0` already implies the last
    // group matched.
    fn finish(self) -> bool {
        self.point_matched && self.count == 0
    }
}

// Accumulator for point within line. Each point is classified interior,
// on boundary or exterior. Boundary overrides interior within a group,
// so the whole group must be scanned (no skip-on-match).
pub(crate) struct PointInLine<'a, K: GeoKernel> {
    points: &'a [K::Point],
    lines: &'a [K::Line],
    current_group: Option<usize>,
    on_interior: bool,
    on_boundary: bool,
    left_to_see: usize,
    pt_inside: bool,
    pt_outside: bool,
    right_side_boundary_qualifies: bool,
}

impl<'a, K: GeoKernel> PointInLine<'a, K> {
    pub(crate) fn new(
        points: &'a [K::Point],
        lines: &'a [K::Line],
        right_side_boundary_qualifies: bool,
    ) -> Self {
        PointInLine {
            left_to_see: points.len(),
            points,
            lines,
            current_group: None,
            on_interior: false,
            on_boundary: false,
            pt_inside: false,
            pt_outside: false,
            right_side_boundary_qualifies,
        }
    }

    fn finalize_group(&mut self) {
        if self.on_boundary {
            // Boundary overrides interior: neither for Within, qualifying only
            // under CoveredBy.
            if self.right_side_boundary_qualifies {
                self.pt_inside = true;
            }
        } else if self.on_interior {
            self.pt_inside = true;
        } else {
            self.pt_outside = true;
        }
    }
}

impl<K: GeoKernel> Accumulator for PointInLine<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if self.current_group.is_some() {
                self.finalize_group();
                if self.pt_outside {
                    return;
                }
            }
            self.left_to_see -= 1;
            self.current_group = Some(li);
            self.on_interior = false;
            self.on_boundary = false;
        }

        let p = &self.points[li];
        let seg = &self.lines[ri];
        if let Some(is_boundary) = point_at_line_endpoint(p, seg) {
            if is_boundary {
                self.on_boundary = true;
            } else {
                self.on_interior = true;
            }
        } else if point_on_segment_t::<K, _>(p, seg).is_some() {
            self.on_interior = true;
        }
    }

    fn ready(&self) -> bool {
        self.pt_outside
    }

    // Within if every point was seen, none exterior, and at least
    // one interior.
    fn finish(mut self) -> bool {
        if self.current_group.is_some() && !self.pt_outside {
            self.finalize_group();
        }
        self.left_to_see == 0 && !self.pt_outside && self.pt_inside
    }
}

// Accumulator for line within line. Every left segment must be fully
// covered by right segments. We collect the covered sub-interval for
// each candidate right segment (in the left segment's own parameter,
// clamped to `[0, 1]`) then at group end sort by start and sweep: the
// union must tile `[0, 1]`. No boundaries here, left is inside or
// outside.
pub(crate) struct LineInLine<'a, K: GeoKernel> {
    left: &'a [K::Line],
    right: &'a [K::Line],
    current_group: Option<usize>,
    group_covered: bool,
    covered_count: usize,
    ranges: Vec<(f64, f64)>,
    line_unmatched: bool,
    // Set once any group is covered; guards the degenerate empty-line case in `finish`.
    line_matched: bool,
}

impl<'a, K: GeoKernel> LineInLine<'a, K> {
    pub(crate) fn new(left: &'a [K::Line], right: &'a [K::Line]) -> Self {
        LineInLine {
            left,
            right,
            current_group: None,
            group_covered: false,
            covered_count: 0,
            ranges: Vec::new(),
            line_unmatched: false,
            line_matched: false,
        }
    }

    fn finalize_group(&mut self) {
        if self.group_covered || fully_covered(&mut self.ranges) {
            self.covered_count += 1;
            self.line_matched = true;
        } else {
            self.line_unmatched = true;
        }
    }
}

// Checks if the union of intervals tile [0, 1]. Sort by start, sweep a
// frontier from 0: every interval must start at or before the running
// frontier (else a gap) and may extend it.
fn fully_covered(ranges: &mut [(f64, f64)]) -> bool {
    if ranges.is_empty() {
        return false;
    }
    ranges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut frontier = 0.0;
    for &(lo, hi) in ranges.iter() {
        if lo > frontier + EPS {
            return false; // gap before this interval starts
        }
        if hi > frontier {
            frontier = hi;
        }
        if frontier >= 1.0 - EPS {
            return true;
        }
    }
    frontier >= 1.0 - EPS
}

impl<K: GeoKernel> Accumulator for LineInLine<'_, K> {
    type K = K;

    // No ray casting for this one, so a symmetric bbox overlap (not the ray
    // pruning) is the correct, tighter prune.
    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        a.box_overlap(b)
    }

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            match self.current_group {
                Some(prev) => {
                    self.finalize_group();
                    if self.line_unmatched || li > prev + 1 {
                        self.line_unmatched = true;
                        return;
                    }
                }
                None if li > 0 => {
                    self.line_unmatched = true;
                    return;
                }
                None => {}
            }
            self.ranges.clear();
            self.group_covered = false;
            self.current_group = Some(li);
        }
        if self.group_covered {
            return;
        }
        if let Some((t_lo, t_hi)) = covered_range::<K, _, _>(&self.left[li], &self.right[ri]) {
            // Early exit if this single right segment already covers the
            // whole left segment.
            if t_lo <= EPS && t_hi >= 1.0 - EPS {
                self.group_covered = true;
            } else {
                self.ranges.push((t_lo, t_hi));
            }
        }
    }

    fn ready(&self) -> bool {
        self.line_unmatched || self.covered_count == self.left.len()
    }

    // like for point x point, the line_matched check is just to make
    // sure the left geo is not empty.
    fn finish(mut self) -> bool {
        if self.current_group.is_some() {
            self.finalize_group();
        }
        self.line_matched && self.covered_count == self.left.len()
    }
}

// Accumulator for line within poly. The poly's edge is a boundary, so a line
// is classified inside, on boundary or outside. Per segment we
// fold two per-edge checks:
// - midpoint_ray_check: ray-cast parity (trustworthy only when the midpoint is off the
//   boundary) plus a midpoint-on-vertex "inside" flag that holds even when parity has
//   been invalidated by another edge.
// - segment_crossing_check: every crossing contributes a signed `t` that must net to
//   zero over the segment; an unbalanced one is a genuine exit to the exterior, while
//   a vertex of another ring touching or an edge interior cancels against the edge-body
//   crossing there. Interior contact is required for the edge to qualify.
pub(crate) struct LineInPoly<'a, K: GeoKernel> {
    left: &'a [K::Line],
    edges: &'a [K::Edge],
    ref_status: &'a RefStatus,
    current_group: Option<usize>,
    next_unseen: usize,
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    s_sum: f64,
    qualified: bool,
    failed: bool,
    any_interior: bool,
    on_boundary_qualified: bool,
    right_side_boundary_qualifies: bool,
}

impl<'a, K: GeoKernel> LineInPoly<'a, K> {
    pub(crate) fn new(
        left: &'a [K::Line],
        edges: &'a [K::Edge],
        ref_status: &'a RefStatus,
        right_side_boundary_qualifies: bool,
    ) -> Self {
        LineInPoly {
            left,
            edges,
            ref_status,
            current_group: None,
            next_unseen: 0,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            s_sum: 0.0,
            qualified: false,
            failed: false,
            any_interior: false,
            on_boundary_qualified: false,
            right_side_boundary_qualifies,
        }
    }

    fn ray_end_inside(&self, li: usize) -> bool {
        !self.ref_status.always_outside()
            && self
                .ref_status
                .inside_for::<K>(self.edges, &K::midpoint(&self.left[li]))
    }

    fn reset_group(&mut self, li: usize) {
        self.current_group = Some(li);
        self.next_unseen = li + 1;
        self.parity = false;
        self.parity_valid = true;
        self.mid_inside = false;
        self.s_sum = 0.0;
        self.qualified = false;
    }

    fn finalize_group(&mut self, li: usize) {
        let inside = self.parity != self.ray_end_inside(li);
        if self.s_sum.abs() > EPS {
            self.failed = true;
        } else if self.mid_inside || self.qualified || (self.parity_valid && inside) {
            self.any_interior = true;
        } else if self.parity_valid && !inside {
            self.failed = true;
        } else {
            // Segment lies entirely on the polygon boundary, neither
            // interior nor exterior.
            self.on_boundary_qualified = true;
        }
    }

    // Segments that matched no edge, never on the boundary.
    fn classify_unseen(&mut self, lis: Range<usize>) {
        if lis.is_empty() {
            return;
        }
        if self.ref_status.always_outside() {
            self.failed = true;
            return;
        }
        for li in lis {
            if self.ray_end_inside(li) {
                self.any_interior = true;
            } else {
                self.failed = true;
                return;
            }
        }
    }
}

impl<K: GeoKernel> Accumulator for LineInPoly<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
            }
            if !self.failed {
                self.classify_unseen(self.next_unseen..li);
            }
            if self.failed {
                return;
            }
            self.reset_group(li);
        }

        let edge = &self.edges[ri];
        let next = &self.edges[edge.next()];
        let seg = &self.left[li];

        let (ray, inside, parity_valid) = midpoint_ray_check::<K, _>(seg, edge, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.mid_inside = true;
        }
        if !parity_valid {
            self.parity_valid = false;
        }

        let (qualifying, signed_t) = segment_crossing_check::<K, _>(seg, edge, next);
        if qualifying {
            self.qualified = true;
        }
        if let Some(t) = signed_t {
            self.s_sum += t;
        }
    }

    fn ready(&self) -> bool {
        self.failed
    }

    // Within if at least one line touches the poly's interior and no line touches
    // the exterior.
    fn finish(mut self) -> bool {
        if self.failed {
            return false;
        }
        if let Some(li) = self.current_group {
            self.finalize_group(li);
        }
        if !self.failed {
            self.classify_unseen(self.next_unseen..self.left.len());
        }
        let qualified =
            self.any_interior || (self.right_side_boundary_qualifies && self.on_boundary_qualified);
        !self.failed && qualified
    }
}

// Accumulator that checks for containment only based on parity checks. This is
// used to check if a right side ring bounding a part of right's complement (a hole,
// or on the sphere any ring) falls inside a left side poly, which is part of the
// check we run to see if left is within right. Such a ring completely contained in a
// left side poly can't really be detected by iterating on left's components, so we
// need to explicitly check for the right ring's containment in left. Since any
// intersections will be caught when checking if left is within right, this
// accumulator here only checks for full containment, i.e. only for ray casting and
// parity. Left ids are relative to `rings_start`.
pub(crate) struct ComplementRingsInLeftPoly<'a, K: GeoKernel> {
    right: &'a [K::Edge],
    rings_start: usize,
    left: &'a [K::Edge],
    left_ref_status: &'a RefStatus,
    current_group: Option<usize>,
    next_unseen: usize,
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    found_inside: bool,
}

impl<'a, K: GeoKernel> ComplementRingsInLeftPoly<'a, K> {
    pub(crate) fn new(
        right: &'a [K::Edge],
        rings_start: usize,
        left: &'a [K::Edge],
        left_ref_status: &'a RefStatus,
    ) -> Self {
        ComplementRingsInLeftPoly {
            right,
            rings_start,
            left,
            left_ref_status,
            current_group: None,
            next_unseen: 0,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            found_inside: false,
        }
    }

    fn ray_end_inside(&self, li: usize) -> bool {
        !self.left_ref_status.always_outside()
            && self
                .left_ref_status
                .inside_for::<K>(self.left, &K::midpoint(&self.right[self.rings_start + li]))
    }

    fn finalize_group(&mut self, li: usize) {
        if self.mid_inside || (self.parity_valid && self.parity != self.ray_end_inside(li)) {
            self.found_inside = true;
        }
    }

    // Ring edges that matched no left edge, inside iff the end of their ray is.
    fn classify_unseen(&mut self, lis: Range<usize>) {
        if self.left_ref_status.always_outside() {
            return;
        }
        let found_inside = lis.into_iter().any(|li| self.ray_end_inside(li));
        self.found_inside = found_inside;
    }
}

impl<K: GeoKernel> Accumulator for ComplementRingsInLeftPoly<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
            }
            if !self.found_inside {
                self.classify_unseen(self.next_unseen..li);
            }
            if self.found_inside {
                return;
            }
            self.current_group = Some(li);
            self.next_unseen = li + 1;
            self.parity = false;
            self.parity_valid = true;
            self.mid_inside = false;
        }
        let ring_edge = &self.right[self.rings_start + li];
        let edge = &self.left[ri];
        let next = &self.left[edge.next()];
        let (ray, inside, parity_valid) = midpoint_ray_check::<K, _>(ring_edge, edge, next);
        if ray {
            self.parity ^= true;
        }
        if inside {
            self.mid_inside = true;
            self.found_inside = true;
        }
        if !parity_valid {
            self.parity_valid = false;
        }
    }

    fn ready(&self) -> bool {
        self.found_inside
    }

    // A single "found inside" is sufficient here, that's really all we care about,
    // if any edge of the right side ring is found inside the left side poly, that's
    // it we can return (true from this accumulator, which will become false for the
    // global check).
    fn finish(mut self) -> bool {
        if let Some(li) = self.current_group {
            self.finalize_group(li);
        }
        if !self.found_inside {
            self.classify_unseen(self.next_unseen..self.right.len() - self.rings_start);
        }
        self.found_inside
    }
}

// Accumulator for the main poly in poly check (which also relies on the complement
// rings in poly accumulator). Checks for intersections to disqualify, since a poly
// can only be contained by a single poly, and for ray casting and parity to qualify.
// We do need to keep track of poly ids here, to make sure that a left poly's edges
// are not contained by multiple right side polys.
pub(crate) struct PolyInPoly<'a, K: GeoKernel> {
    left: &'a [K::Edge],
    left_poly_ids: &'a [usize],
    right: &'a [K::Edge],
    right_poly_ids: &'a [usize],
    right_ref_status: &'a RefStatus,
    current_group: Option<usize>,
    next_unseen: usize,
    current_left_poly: Option<usize>,
    container: Option<usize>,
    parity: bool,
    parity_valid: bool,
    mid_inside: bool,
    qualified: bool,
    edge_container: Option<(f64, usize)>,
    failed: bool,
}

impl<'a, K: GeoKernel> PolyInPoly<'a, K> {
    pub(crate) fn new(
        left: &'a [K::Edge],
        left_poly_ids: &'a [usize],
        right: &'a [K::Edge],
        right_poly_ids: &'a [usize],
        right_ref_status: &'a RefStatus,
    ) -> Self {
        PolyInPoly {
            left,
            left_poly_ids,
            right,
            right_poly_ids,
            right_ref_status,
            current_group: None,
            next_unseen: 0,
            current_left_poly: None,
            container: None,
            parity: false,
            parity_valid: true,
            mid_inside: false,
            qualified: false,
            edge_container: None,
            failed: false,
        }
    }

    // The right poly the end of the ray cast from the left edge's midpoint is in.
    fn ray_end_poly(&self, li: usize) -> Option<usize> {
        if self.right_ref_status.always_outside() {
            return None;
        }
        self.right_ref_status
            .containing_poly_for::<K>(self.right, &K::midpoint(&self.left[li]))
    }

    fn reset_group(&mut self, li: usize) {
        self.current_group = Some(li);
        self.next_unseen = li + 1;
        self.parity = false;
        self.parity_valid = true;
        self.mid_inside = false;
        self.qualified = false;
        self.edge_container = None;
    }

    // This is to keep track of which right side poly the edge closest to
    // the left side edge belongs to. Since we need to know which right side
    // poly contains the left side edges (to detect the case where multiple
    // right side polys contain the left poly), we need to keep track of
    // which edge is closest to the left edge so that if parity indicates
    // the left edge is contained, we know which right side poly it's contained
    // in, and we can track across other edges from the same left side poly.
    fn consider_container(&mut self, dist: f64, rp: usize) {
        if self.edge_container.is_none_or(|(d, _)| dist < d) {
            self.edge_container = Some((dist, rp));
        }
    }

    // A left edge contained in the right poly c must have the same container as the
    // other edges of its left poly.
    fn track_container(&mut self, li: usize, c: usize) {
        let lp = self.left_poly_ids[li];
        if self.current_left_poly != Some(lp) {
            self.current_left_poly = Some(lp);
            self.container = None;
        }
        match self.container {
            None => self.container = Some(c),
            Some(q) if q != c => self.failed = true,
            Some(_) => {}
        }
    }

    fn finalize_group(&mut self, li: usize) {
        let ray_end_poly = self.ray_end_poly(li);
        self.qualified |=
            (self.parity_valid && self.parity != ray_end_poly.is_some()) || self.mid_inside;
        if !self.qualified {
            self.failed = true; // a strictly-exterior edge
            return;
        }
        // The ray leaves the container before reaching any other poly, so when it
        // crosses anything the nearest crossing is the container's. A qualified edge
        // without crossings is in the poly the end of its ray is in.
        let c = self
            .edge_container
            .map(|(_, c)| c)
            .or(ray_end_poly)
            .expect("qualified edge has a container");
        self.track_container(li, c);
    }

    // Edges that matched no right edge, inside iff the end of their ray is, in the
    // poly the end is in.
    fn classify_unseen(&mut self, lis: Range<usize>) {
        if lis.is_empty() {
            return;
        }
        if self.right_ref_status.always_outside() {
            self.failed = true;
            return;
        }
        for li in lis {
            match self.ray_end_poly(li) {
                Some(c) => self.track_container(li, c),
                None => self.failed = true,
            }
            if self.failed {
                return;
            }
        }
    }
}

impl<K: GeoKernel> Accumulator for PolyInPoly<'_, K> {
    type K = K;

    fn update(&mut self, li: usize, ri: usize) {
        if self.current_group != Some(li) {
            if let Some(prev) = self.current_group {
                self.finalize_group(prev);
            }
            if !self.failed {
                self.classify_unseen(self.next_unseen..li);
            }
            if self.failed {
                return;
            }
            self.reset_group(li);
        }

        let edge = &self.right[ri];
        let next = &self.right[edge.next()];
        let seg = &self.left[li];
        let rp = self.right_poly_ids[ri];

        let (ray, inside, parity_valid) = midpoint_ray_check::<K, _>(seg, edge, next);
        if ray {
            self.parity ^= true;
            if let Some(dist) = K::ray_crossing_distance(&K::midpoint(seg), edge) {
                self.consider_container(dist, rp);
            }
        }
        if inside {
            self.mid_inside = true;
            self.consider_container(0.0, rp);
        }
        if !parity_valid {
            self.parity_valid = false;
        }

        let (disqualifies, collinear_inside) = poly_edge_relation::<K>(seg, edge, next);
        if disqualifies {
            self.failed = true;
        }
        if collinear_inside {
            self.qualified = true;
            self.consider_container(0.0, rp);
        }
    }

    fn ready(&self) -> bool {
        self.failed
    }

    // Within if no edge disqualified, and the left poly was non-empty.
    fn finish(mut self) -> bool {
        if self.failed {
            return false;
        }
        if let Some(li) = self.current_group {
            self.finalize_group(li);
        }
        if !self.failed {
            self.classify_unseen(self.next_unseen..self.left.len());
        }
        !self.failed && !self.left.is_empty()
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    use super::{st_contains, st_within};
    use crate::geospatial::geos::joinable_geo::{JoinableGeo, Planar, Spherical};
    pub(super) use crate::geospatial::test_utils::wkt_to_wkb;

    pub(super) fn compare_within(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let mut geos_result = geos_a.within(&geos_b).unwrap();

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();
        let mut joinable_result = st_within(&joinable_a, &joinable_b);

        assert_eq!(geos_result, joinable_result, "{label}");

        geos_result = geos_b.contains(&geos_a).unwrap();
        joinable_result = st_contains(&joinable_b, &joinable_a);

        assert_eq!(geos_result, joinable_result, "{label}, st_contains version");
    }

    // No oracle for geographies, so the expected result is given.
    pub(super) fn check_geography_within(label: &str, wkt_a: &str, wkt_b: &str, expected: bool) {
        let joinable_a = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_a)).unwrap();
        let joinable_b = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_b)).unwrap();

        assert_eq!(expected, st_within(&joinable_a, &joinable_b), "{label}");
        assert_eq!(
            expected,
            st_contains(&joinable_b, &joinable_a),
            "{label}, st_contains version"
        );
    }
}

#[cfg(test)]
mod geography_within_tests {
    use super::test_helpers::check_geography_within;

    // - big_square contains small_square, and overlapping_square overlaps
    //   big_square without either containing the other.
    // - cap_70: a cap around the north pole, its ring at latitude 70 going east.
    // - all_but_cap_80: its ring at latitude 80 goes west, so it's everything but
    //   the cap above it, most of the globe. Its edges are inside cap_70, but
    //   cap_70's shell is inside it, so it's not within cap_70. That's only caught
    //   by checking cap_70's shell against it, not just its holes (it has none).
    #[test]
    fn poly_poly() {
        let big_square = "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))";
        let small_square = "POLYGON ((12 12, 14 12, 14 14, 12 14, 12 12))";
        let overlapping_square = "POLYGON ((15 15, 25 15, 25 25, 15 25, 15 15))";
        let cap_70 = "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70))";
        let all_but_cap_80 = "POLYGON ((0 80, -90 80, 180 80, 90 80, 0 80))";

        check_geography_within("small square in big square", small_square, big_square, true);
        check_geography_within(
            "big square in small square",
            big_square,
            small_square,
            false,
        );

        check_geography_within("overlapping squares", overlapping_square, big_square, false);
        check_geography_within(
            "overlapping squares, swapped",
            big_square,
            overlapping_square,
            false,
        );

        check_geography_within("most of the globe in cap", all_but_cap_80, cap_70, false);
        check_geography_within("cap in most of the globe", cap_70, all_but_cap_80, false);
        check_geography_within(
            "small square in most of the globe",
            small_square,
            all_but_cap_80,
            true,
        );
    }

    // Containment in polys containing the north pole, where rays reach the pole
    // inside the poly without crossing anything, so whether something is inside,
    // and in which poly, comes from where the end of its ray is.
    // - cap_70: a cap around the north pole, its ring at latitude 70 going east,
    //   its edges peaking at 75.57. cap_70_small_hole has a hole whose edges are
    //   ray candidates for polar_square's top edge, but aren't crossed by its ray.
    // - cap_70_pole_hole: cap_70 with a hole around the pole, its ring at latitude
    //   80 going west, its edges peaking at 82.89.
    // - pole_triangle: a triangle with a vertex at the pole, its sides along the
    //   meridians 0 and 90. The diamonds each contain one of its edges, touching
    //   each other only at its vertices, two of them at the pole. Each edge is
    //   inside, but in a different diamond, so it's not within them. The diamonds'
    //   shells going through the triangle, that's caught by checking them against
    //   it, before the edges' containers are compared.
    // - wedges: two wedges with a vertex at the pole, between the meridians 0 and
    //   80, and 100 and 180, touching only at the pole. The squares' rays reach the
    //   pole without crossing anything, so which wedge they're in depends on the
    //   direction they reach it from, the one in the gap between the wedges being
    //   in neither.
    #[test]
    fn pole_containment() {
        let cap_70 = "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70))";
        let cap_70_small_hole =
            "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70), (35 87, 35 88, 36 88, 36 87, 35 87))";
        let cap_70_pole_hole = "POLYGON ((0 70, 90 70, 180 70, -90 70, 0 70), \
            (0 80, -90 80, 180 80, 90 80, 0 80))";
        let polar_square = "POLYGON ((30 84, 50 84, 50 86, 30 86, 30 84))";
        let pole_triangle = "POLYGON ((0 80, 90 80, 0 90, 0 80))";
        let diamonds = "MULTIPOLYGON (\
            ((0 80, 5 85, 0 90, -5 85, 0 80)), \
            ((90 80, 95 85, 0 90, 85 85, 90 80)), \
            ((0 80, 45 76, 90 80, 45 86, 0 80)))";
        let wedges = "MULTIPOLYGON (((0 70, 80 70, 0 90, 0 70)), ((100 70, 180 70, 0 90, 100 70)))";
        let square_in_each_wedge = "MULTIPOLYGON (\
            ((30 84, 50 84, 50 86, 30 86, 30 84)), \
            ((130 84, 150 84, 150 86, 130 86, 130 84)))";
        let square_between_wedges = "POLYGON ((85 84, 95 84, 95 86, 85 86, 85 84))";

        check_geography_within("polar square in cap", polar_square, cap_70, true);
        check_geography_within(
            "polar square in cap with small hole",
            polar_square,
            cap_70_small_hole,
            true,
        );
        check_geography_within("pole triangle in cap", pole_triangle, cap_70, true);
        check_geography_within("pole triangle in diamonds", pole_triangle, diamonds, false);
        check_geography_within("a square in each wedge", square_in_each_wedge, wedges, true);
        check_geography_within(
            "square between the wedges",
            square_between_wedges,
            wedges,
            false,
        );

        check_geography_within(
            "points in cap, matching no edge",
            "MULTIPOINT (45 80, 135 85)",
            cap_70,
            true,
        );
        check_geography_within(
            "points in cap, one outside",
            "MULTIPOINT (45 80, 45 60)",
            cap_70,
            false,
        );
        check_geography_within(
            "points in cap with pole hole, one in the hole matching no edge",
            "MULTIPOINT (45 78, 45 88)",
            cap_70_pole_hole,
            false,
        );

        check_geography_within(
            "line in cap, matching no edge",
            "LINESTRING (30 85, 60 85)",
            cap_70,
            true,
        );
        check_geography_within(
            "line leaving the cap",
            "LINESTRING (45 80, 45 60)",
            cap_70,
            false,
        );
    }
}

#[cfg(test)]
mod within_tests {
    use super::test_helpers::compare_within;

    #[test]
    fn point_left() {
        compare_within("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_within("point left, case 2", "POINT (1 1)", "POINT (2 2)");
        compare_within("point left, case 3", "POINT (1 1)", "MULTIPOINT (0 0, 1 1)");
        compare_within("point left, case 4", "POINT (1 1)", "MULTIPOINT (0 0, 2 2)");

        compare_within(
            "point left, case 5",
            "POINT (0.5 0)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_within("point left, case 6", "POINT (2 0)", "LINESTRING (0 0, 1 0)");
        compare_within("point left, case 7", "POINT (0 0)", "LINESTRING (0 0, 1 0)");

        compare_within(
            "point left, case 8",
            "POINT (1 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_within(
            "point left, case 9",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );

        compare_within(
            "point left, case 10",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
        );

        compare_within(
            "point left, case 11",
            "POINT (0 1)",
            "LINESTRING (0 3, 0 0, 1 0, 0 1)",
        );

        compare_within(
            "point left, case 12",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2)",
        );

        compare_within(
            "point left, case 13",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, -1 2)",
        );

        compare_within(
            "point left, case 14",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2, -1 2)",
        );

        compare_within(
            "point left, case 15",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 2 2), (0 2, 2 0))",
        );

        compare_within(
            "point left, case 16",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0))",
        );

        compare_within(
            "point left, case 17",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0), (1 1, 1 2))",
        );

        compare_within(
            "point left, case 18",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1, 2 2), (0 2, 1 1, 2 0))",
        );

        compare_within(
            "point left, case 19",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 20",
            "POINT (3 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 21",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 22",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 23",
            "POINT (0.5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 24",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 25",
            "POINT (1 1)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 26",
            "POINT (1 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_within(
            "point left, case 27",
            "POINT (-1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 28",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 29",
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 30",
            "POINT (2 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_within(
            "point left, case 31",
            "POINT (2 3)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_within(
            "point left, case 32",
            "POINT (2 1)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_within(
            "point left, case 33",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_within(
            "point left, case 34",
            "POINT (2.5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_within(
            "point left, case 35",
            "POINT (2 0)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_within(
            "point left, case 36",
            "POINT (-1 4)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );

        compare_within(
            "point left, case 37",
            "POINT (1 2)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_within(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (1 1)",
        );

        compare_within(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_within(
            "multi point left, case 3",
            "MULTIPOINT (1 1, 2 2, 3 3)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_within(
            "multi point left, case 4",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2, 3 3)",
        );

        compare_within(
            "multi point left, case 5",
            "MULTIPOINT (0 1, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 6",
            "MULTIPOINT (1 0, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 7",
            "MULTIPOINT (0.5 0, 1.5 0)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 8",
            "MULTIPOINT (1 0, 0 0)",
            "LINESTRING (0 0, 2 0)",
        );

        compare_within(
            "multi point left, case 9",
            "MULTIPOINT (2 0, 1 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "multi point left, case 10",
            "MULTIPOINT (0 0, 2 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "multi point left, case 11",
            "MULTIPOINT (1 0, 2 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );

        compare_within(
            "multi point left, case 12",
            "MULTIPOINT (0 0, 3 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );

        compare_within(
            "multi point left, case 13",
            "MULTIPOINT (2 2, 0 0)",
            "MULTILINESTRING ((0 0, 2 2), (4 0, 2 2), (2 4, 2 2))",
        );

        compare_within(
            "multi point left, case 14",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0))",
        );

        compare_within(
            "multi point left, case 15",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0), (2 0, 2 2))",
        );

        compare_within(
            "multi point left, case 16",
            "MULTIPOINT (-1 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 17",
            "MULTIPOINT (2 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 18",
            "MULTIPOINT (0 2, 4 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 19",
            "MULTIPOINT (0 2, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 20",
            "MULTIPOINT (0 0, 4 4)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 21",
            "MULTIPOINT (0 0, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "multi point left, case 22",
            "MULTIPOINT (3 5, 5 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 23",
            "MULTIPOINT (3 3, 7 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 24",
            "MULTIPOINT (5 5, 1 1)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 25",
            "MULTIPOINT (1 1, 9 9)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 26",
            "MULTIPOINT (1 3, 1 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "multi point left, case 27",
            "MULTIPOINT (-1 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_within(
            "multi point left, case 28",
            "MULTIPOINT (2 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_within(
            "multi point left, case 29",
            "MULTIPOINT (2 2, 8 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_within("line left, case 1", "LINESTRING (0 0, 4 0)", "POINT (2 0)");

        compare_within(
            "line left, case 2",
            "LINESTRING (0 0, 4 0)",
            "MULTIPOINT (1 0, 3 0)",
        );

        compare_within(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 0, 4 0)",
        );

        compare_within(
            "line left, case 4",
            "LINESTRING (0 0, 4 4)",
            "LINESTRING (0 4, 4 0)",
        );

        compare_within(
            "line left, case 5",
            "LINESTRING (0 0, 2 2, 4 0)",
            "LINESTRING (0 4, 2 2, 4 4)",
        );

        compare_within(
            "line left, case 6",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (5 0, 7 0)",
        );

        compare_within(
            "line left, case 7",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (3 0, 5 0)",
        );

        compare_within(
            "line left, case 8",
            "LINESTRING (0 0, 3 0)",
            "LINESTRING (2 0, 5 0)",
        );

        compare_within(
            "line left, case 9",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );

        compare_within(
            "line left, case 10",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (1 0, 3 0)",
        );

        compare_within(
            "line left, case 11",
            "LINESTRING (0 0, 4 0, 2 2)",
            "LINESTRING (0 0, 4 0, 4 4)",
        );

        compare_within(
            "line left, case 12",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
        );

        compare_within(
            "line left, case 13",
            "LINESTRING (0 2, 4 2)",
            "LINESTRING (0 0, 2 2, 4 0)",
        );

        compare_within(
            "line left, case 14",
            "LINESTRING (0 0, 2 0, 2 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 2 2), (2 2, 4 2))",
        );

        compare_within(
            "line left, case 15",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_within(
            "line left, case 16",
            "LINESTRING (1 0, 3 0)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
        );

        compare_within(
            "line left, case 17",
            "LINESTRING (0 0, 4 0)",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "line left, case 18",
            "LINESTRING (0 0, 2 0, 2 2)",
            "MULTILINESTRING ((-1 0, 3 0), (2 -1, 2 3))",
        );

        compare_within(
            "line left, case 19",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0), (4 0, 6 0))",
        );

        compare_within(
            "line left, case 20",
            "LINESTRING (1 0, 5 0)",
            "LINESTRING (0 0, 4 0, 4 2, 5 2, 5 0, 2 0)",
        );

        compare_within(
            "line left, case 21",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0))",
        );

        compare_within(
            "line left, case 18",
            "LINESTRING (-2 3, -1 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 19",
            "LINESTRING (1 3, 5 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 20",
            "LINESTRING (-1 3, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 21",
            "LINESTRING (-2 -2, 2 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 22",
            "LINESTRING (-1 0, 3 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 23",
            "LINESTRING (1 0, 5 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 24",
            "LINESTRING (0 0, 6 0, 6 6, 0 6, 0 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 25",
            "LINESTRING (1 5, 2 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 26",
            "LINESTRING (4 5, 6 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 27",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 28",
            "LINESTRING (1 1, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 29",
            "LINESTRING (3 3, 3 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "line left, case 30",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)))",
        );

        compare_within(
            "line left, case 31",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)), ((2 2, 1 4, 3 4, 2 2)))",
        );

        compare_within(
            "line left, case 32",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_within(
            "line left, case 33",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_within(
            "line left, case 34",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 35",
            "LINESTRING (-1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 36",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 37",
            "LINESTRING (-1 5, 9 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 38",
            "LINESTRING (1 5, 7 5)",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );

        compare_within(
            "line left, case 39",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 40",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 41",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 42",
            "LINESTRING (0 -2, 0 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_within(
            "line left, case 43",
            "LINESTRING (-1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 44",
            "LINESTRING (1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 45",
            "LINESTRING (6 4, -1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 46",
            "LINESTRING (6 4, 1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "line left, case 47",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 2, 4 1, 3 0, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_within(
            "line left, case 48",
            "LINESTRING (0.5 2, 4 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_within(
            "line left, case 49",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_within(
            "line left, case 50",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 0 4, 2 2, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_within(
            "line left, case 51",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 0 8, 4 5, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 52",
            "LINESTRING (5 5, 1 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 53",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 54",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_within(
            "line left, case 55",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_within(
            "line left, case 56",
            "LINESTRING (1 4, 7 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 57",
            "LINESTRING (0 4, 5.5 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 58",
            "LINESTRING (1 2, 7 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 59",
            "LINESTRING (0 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_within(
            "line left, case 60",
            "LINESTRING (0 2, 8 2)",
            "POLYGON ((4 0, 8 0, 8 4, 4 4, 4 0))",
        );

        compare_within(
            "line left, case 61",
            "LINESTRING (0.5 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 2, 3 1, 4 2, 3 3, 2 2)))",
        );

        compare_within(
            "line left, case 62",
            "LINESTRING (1 1, 5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)))",
        );

        compare_within(
            "line left, case 63",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 4 6, 3 6, 3 8, 0 8, 0 0))",
        );
    }

    #[test]
    fn multi_line_left() {
        compare_within(
            "multi line left, case 1",
            "MULTILINESTRING ((1 0, 3 0), (1 2, 3 2))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 2",
            "MULTILINESTRING ((1 0, 3 0), (1 1, 3 1))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 3",
            "MULTILINESTRING ((1 1, 3 1), (1 3, 3 3))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_within(
            "multi line left, case 4",
            "MULTILINESTRING ((1 1, 3 1), (6 1, 8 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 5",
            "MULTILINESTRING ((1 1, 3 1), (10 1, 12 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 6",
            "MULTILINESTRING ((10 1, 12 1), (10 3, 12 3))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_within(
            "multi line left, case 7",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
            "LINESTRING (0 0, 2 0, 4 0)",
        );
    }

    #[test]
    fn poly_left() {
        compare_within(
            "poly left, case 1",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POINT (2 2)",
        );

        compare_within(
            "poly left, case 2",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "LINESTRING (0 0, 4 4)",
        );

        compare_within(
            "poly left, case 3",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 4",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 5",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 6",
            "POLYGON ((5 5, 9 5, 9 9, 5 9, 5 5))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_within(
            "poly left, case 7",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 8",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1), (2 2, 2 8, 8 8, 8 2, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 9",
            "POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 10",
            "POLYGON ((2 2, 5 2, 5 5, 2 5, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_within(
            "poly left, case 11",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 2, 4 2, 4 6, 0 6, 0 2))",
        );

        compare_within(
            "poly left, case 12",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
            "MULTIPOLYGON (((0 2, 2 2, 2 4, 0 4, 0 2)), ((2 4, 4 4, 4 6, 2 6, 2 4)), ((4 2, 6 2, 6 4, 4 4, 4 2)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_within(
            "poly left, case 13",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "MULTIPOLYGON (
                ((1 1, 2 0.5, 3 1, 2 1.5, 1 1)),
                ((3 1, 3.5 2, 3 3, 2.5 2, 3 1)),
                ((3 3, 2 3.5, 1 3, 2 2.5, 3 3)),
                ((1 3, 0.5 2, 1 1, 1.5 2, 1 3)))
            ",
        );

        compare_within(
            "poly left, case 14",
            "POLYGON ((1 4, 7 4, 7 5, 1 5, 1 4))",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_within(
            "multi poly left, case 1",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((3 1, 4 1, 4 2, 3 2, 3 1)))",
            "POLYGON ((0 0, 5 0, 5 3, 0 3, 0 0))",
        );

        compare_within(
            "multi poly left, case 2",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_within(
            "multi poly left, case 3",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)), ((10 0, 14 0, 14 3, 10 3, 10 0)))",
        );

        compare_within(
            "multi poly left, case 4",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );

        compare_within(
            "multi poly left, case 5",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((10 1, 11 1, 11 2, 10 2, 10 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_within(
            "multi poly left, case 6",
            "MULTIPOLYGON (((6 1, 7 1, 7 2, 6 2, 6 1)), ((8 1, 9 1, 9 2, 8 2, 8 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );
    }

    #[test]
    fn empty_geo() {
        let sq = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        // empty on the left
        compare_within("empty multipoint left", "MULTIPOINT EMPTY", sq);
        compare_within("empty multilinestring left", "MULTILINESTRING EMPTY", sq);
        compare_within("empty polygon left", "POLYGON EMPTY", sq);
        // empty on the right
        compare_within("empty multipoint right", sq, "MULTIPOINT EMPTY");
        compare_within("empty multilinestring right", sq, "MULTILINESTRING EMPTY");
        compare_within("empty polygon right", sq, "POLYGON EMPTY");
    }
}
