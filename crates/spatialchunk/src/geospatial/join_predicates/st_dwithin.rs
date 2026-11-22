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

use std::marker::PhantomData;

use super::st_intersects::st_intersects;
use crate::geospatial::geos::joinable_geo::{
    Accumulator, Bboxable, GeoKernel, JoinableGeo, Segment,
};

// *************************************************************
// dwithin(a, b, distance) -> true if the minimum distance between a
// and b is <= distance.
// *************************************************************
pub(crate) fn st_dwithin<K: GeoKernel>(
    a: &JoinableGeo<K>,
    b: &JoinableGeo<K>,
    distance: f64,
) -> bool {
    match (a, b) {
        (JoinableGeo::Point { points: a_pts, .. }, JoinableGeo::Point { points: b_pts, .. }) => {
            let mut acc = PointToPointDist::<K>::new(a_pts, b_pts, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points, .. }, JoinableGeo::Line { lines, .. }) => {
            let mut acc = PointToSegmentsDist::<K, _>::new(points, lines, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Point { points, .. }, JoinableGeo::Poly { edges, .. }) => {
            // A point inside the polygon has distance 0 but positive distance to
            // every edge, so the containment case needs an explicit intersects
            // check.
            if distance >= 0.0 && a.top_bbox_overlaps(b) && st_intersects(a, b) {
                return true;
            }
            let mut acc = PointToSegmentsDist::<K, _>::new(points, edges, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            // Lines have no interior, so distance 0 means they cross, that needs an
            // explicit intersects check. Otherwise no pair crosses, so the segment
            // distance below may assume disjoint pairs.
            if distance >= 0.0 && a.top_bbox_overlaps(b) && st_intersects(a, b) {
                return true;
            }
            let mut acc = SegmentsToSegmentsDist::<K, _, _>::new(l_lines, r_lines, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Line { lines, .. }, JoinableGeo::Poly { edges, .. }) => {
            // A line inside the polygon (or crossing it) has distance 0, that needs
            // an explicit intersects check. Otherwise no pair crosses, so the
            // segment distance below may assume disjoint pairs.
            if distance >= 0.0 && a.top_bbox_overlaps(b) && st_intersects(a, b) {
                return true;
            }
            let mut acc = SegmentsToSegmentsDist::<K, _, _>::new(lines, edges, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        (JoinableGeo::Poly { edges: a_edges, .. }, JoinableGeo::Poly { edges: b_edges, .. }) => {
            // One polygon inside the other (or crossing boundaries) has distance 0,
            // that needs an explicit intersects check. Otherwise no pair crosses,
            // so the segment distance below may assume disjoint pairs.
            if distance >= 0.0 && a.top_bbox_overlaps(b) && st_intersects(a, b) {
                return true;
            }
            let mut acc = SegmentsToSegmentsDist::<K, _, _>::new(a_edges, b_edges, distance);
            a.fold_for_unordered_check(b, &mut acc);
            acc.finish()
        }

        // dwithin is symmetric; reuse the arms above with the operands swapped.
        (JoinableGeo::Line { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => st_dwithin(b, a, distance),
    }
}

// Converts a distance threshold to the kernel's internal squared measure, which the
// distances are compared in. A negative threshold can never be satisfied, so it
// maps to -inf.
fn threshold_sq<K: GeoKernel>(distance: f64) -> f64 {
    if distance < 0.0 {
        f64::NEG_INFINITY
    } else {
        K::to_internal_dist_sq(distance)
    }
}

pub(crate) struct PointToPointDist<'a, K: GeoKernel> {
    left: &'a [K::Point],
    right: &'a [K::Point],
    threshold_sq: f64,
    min_dist_sq: f64,
}

impl<'a, K: GeoKernel> PointToPointDist<'a, K> {
    pub(crate) fn new(left: &'a [K::Point], right: &'a [K::Point], distance: f64) -> Self {
        PointToPointDist {
            left,
            right,
            threshold_sq: threshold_sq::<K>(distance),
            min_dist_sq: f64::INFINITY,
        }
    }
}

impl<K: GeoKernel> Accumulator for PointToPointDist<'_, K> {
    type K = K;

    // Points carry no meaningful box for a distance prune, so consider every pair.
    fn prune(&self, _a: &impl Bboxable, _b: &impl Bboxable) -> bool {
        true
    }

    fn update(&mut self, li: usize, ri: usize) {
        let d = K::dist_sq(&self.left[li], &self.right[ri]);
        if d < self.min_dist_sq {
            self.min_dist_sq = d;
        }
    }

    fn ready(&self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }

    fn finish(self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }
}

// The point-to-segment distance captures the distance-zero (point on
// a segment) case. A point inside a polygon is handled separately by
// the caller.
pub(crate) struct PointToSegmentsDist<'a, K: GeoKernel, S> {
    points: &'a [K::Point],
    segs: &'a [S],
    threshold_sq: f64,
    min_dist_sq: f64,
}

impl<'a, K: GeoKernel, S> PointToSegmentsDist<'a, K, S> {
    pub(crate) fn new(points: &'a [K::Point], segs: &'a [S], distance: f64) -> Self {
        PointToSegmentsDist {
            points,
            segs,
            threshold_sq: threshold_sq::<K>(distance),
            min_dist_sq: f64::INFINITY,
        }
    }
}

impl<K: GeoKernel, S: Segment<P = K::Point>> Accumulator for PointToSegmentsDist<'_, K, S> {
    type K = K;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        K::box_min_dist_sq(&a.bbox(), &b.bbox()) <= self.threshold_sq
    }

    fn update(&mut self, li: usize, ri: usize) {
        let d = K::dist_sq_to_segment(&self.points[li], &self.segs[ri]);
        if d < self.min_dist_sq {
            self.min_dist_sq = d;
        }
    }

    fn ready(&self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }

    fn finish(self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }
}

// Uses `dist_sq_to_disjoint`, which assumes the pair does not cross,
// the caller only reaches this after establishing non-intersection.
pub(crate) struct SegmentsToSegmentsDist<'a, K: GeoKernel, L, R> {
    left: &'a [L],
    right: &'a [R],
    threshold_sq: f64,
    min_dist_sq: f64,
    kernel: PhantomData<K>,
}

impl<'a, K: GeoKernel, L, R> SegmentsToSegmentsDist<'a, K, L, R> {
    pub(crate) fn new(left: &'a [L], right: &'a [R], distance: f64) -> Self {
        SegmentsToSegmentsDist {
            left,
            right,
            threshold_sq: threshold_sq::<K>(distance),
            min_dist_sq: f64::INFINITY,
            kernel: PhantomData,
        }
    }
}

impl<K: GeoKernel, L: Segment<P = K::Point>, R: Segment<P = K::Point>> Accumulator
    for SegmentsToSegmentsDist<'_, K, L, R>
{
    type K = K;

    fn prune(&self, a: &impl Bboxable, b: &impl Bboxable) -> bool {
        K::box_min_dist_sq(&a.bbox(), &b.bbox()) <= self.threshold_sq
    }

    fn update(&mut self, li: usize, ri: usize) {
        let d = K::dist_sq_to_disjoint(&self.left[li], &self.right[ri]);
        if d < self.min_dist_sq {
            self.min_dist_sq = d;
        }
    }

    fn ready(&self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }

    fn finish(self) -> bool {
        self.min_dist_sq <= self.threshold_sq
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    use super::st_dwithin;
    use crate::geospatial::geos::joinable_geo::{JoinableGeo, Planar, Spherical};
    pub(super) use crate::geospatial::test_utils::wkt_to_wkb;

    // Checks st_dwithin against GEOS (min distance <= threshold), both operand
    // orders since dwithin is symmetric.
    pub(super) fn compare_dwithin(label: &str, wkt_a: &str, wkt_b: &str, distance: f64) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.distance(&geos_b).unwrap() <= distance;

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();

        assert_eq!(
            geos_result,
            st_dwithin(&joinable_a, &joinable_b, distance),
            "{label}"
        );
        assert_eq!(
            geos_result,
            st_dwithin(&joinable_b, &joinable_a, distance),
            "{label}, reversed"
        );
    }

    // No oracle for geographies, so the expected result is given. distance is in
    // meters.
    pub(super) fn check_geography_dwithin(
        label: &str,
        wkt_a: &str,
        wkt_b: &str,
        distance: f64,
        expected: bool,
    ) {
        let joinable_a = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_a)).unwrap();
        let joinable_b = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_b)).unwrap();

        assert_eq!(
            expected,
            st_dwithin(&joinable_a, &joinable_b, distance),
            "{label}"
        );
        assert_eq!(
            expected,
            st_dwithin(&joinable_b, &joinable_a, distance),
            "{label}, reversed"
        );
    }
}

#[cfg(test)]
mod geography_dwithin_tests {
    use super::test_helpers::check_geography_dwithin;

    // square's right side is along the meridian 10 and next_square's left side
    // along the meridian 11, so they're closest at their southern end, latitude 2,
    // where a degree of longitude is about 111 km.
    #[test]
    fn poly_poly() {
        let square = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
        let square_inside = "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))";
        let next_square = "POLYGON ((11 2, 13 2, 13 4, 11 4, 11 2))";

        check_geography_dwithin("square in square, 0 m", square_inside, square, 0.0, true);
        check_geography_dwithin(
            "squares 1 degree apart, 200 km",
            next_square,
            square,
            200_000.0,
            true,
        );
        check_geography_dwithin(
            "squares 1 degree apart, 50 km",
            next_square,
            square,
            50_000.0,
            false,
        );
    }

    // Lines, unlike points, are pruned by the distance between their bboxes.
    // - Across the antimeridian, 1 degree apart along the equator (111.2 km).
    // - Across the north pole, 2 degrees apart on the meridians 0 and 180 (222.4
    //   km), their longitudes 180 degrees apart.
    // - A point 0.21 degrees above the peak of an edge bulging to 67.79 (23.1 km),
    //   8 degrees above its endpoints.
    #[test]
    fn lon_lat_distortions() {
        let west_of_antimeridian = "LINESTRING (178.5 0, 179.5 0)";
        let east_of_antimeridian = "LINESTRING (-179.5 0, -178.5 0)";
        let meridian_0 = "LINESTRING (0 88, 0 89)";
        let meridian_180 = "LINESTRING (180 88, 180 89)";
        let bulging_edge = "LINESTRING (0 60, 90 60)";

        check_geography_dwithin(
            "across the antimeridian, 200 km",
            west_of_antimeridian,
            east_of_antimeridian,
            200_000.0,
            true,
        );
        check_geography_dwithin(
            "across the antimeridian, 50 km",
            west_of_antimeridian,
            east_of_antimeridian,
            50_000.0,
            false,
        );
        check_geography_dwithin(
            "across the pole, 250 km",
            meridian_0,
            meridian_180,
            250_000.0,
            true,
        );
        check_geography_dwithin(
            "across the pole, 200 km",
            meridian_0,
            meridian_180,
            200_000.0,
            false,
        );
        check_geography_dwithin(
            "above the bulge, 50 km",
            "POINT (45 68)",
            bulging_edge,
            50_000.0,
            true,
        );
        check_geography_dwithin(
            "above the bulge, 10 km",
            "POINT (45 68)",
            bulging_edge,
            10_000.0,
            false,
        );
    }
}

#[cfg(test)]
mod dwithin_tests {
    use super::test_helpers::compare_dwithin;

    #[test]
    fn point_left() {
        compare_dwithin(
            "point x point, out of range",
            "POINT (0 0)",
            "POINT (5 0)",
            2.0,
        );
        compare_dwithin("point x point, in range", "POINT (0 0)", "POINT (1 0)", 2.0);
        compare_dwithin("point x point, equal", "POINT (0 0)", "POINT (0 0)", 0.01);

        compare_dwithin(
            "point on line",
            "POINT (1 0)",
            "LINESTRING (0 0, 2 0)",
            0.01,
        );
        compare_dwithin(
            "point on line endpoint",
            "POINT (0 0)",
            "LINESTRING (0 0, 2 0)",
            0.01,
        );
        compare_dwithin(
            "point in range of line",
            "POINT (1 1)",
            "LINESTRING (0 0, 2 0)",
            2.0,
        );
        compare_dwithin(
            "point out of range of line",
            "POINT (1 5)",
            "LINESTRING (0 0, 2 0)",
            2.0,
        );

        compare_dwithin(
            "point in range of one multiline segment",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 2 0), (0 10, 2 10))",
            2.0,
        );
        compare_dwithin(
            "point out of range of multiline segments",
            "POINT (1 5)",
            "MULTILINESTRING ((0 0, 2 0), (0 10, 2 10))",
            2.0,
        );

        compare_dwithin(
            "point in poly",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "point on poly edge",
            "POINT (0 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "point on poly vertex",
            "POINT (0 0)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "point in range of poly",
            "POINT (5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        compare_dwithin(
            "point out of range of poly",
            "POINT (10 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );

        compare_dwithin(
            "point in hole, out of range of hole edges",
            "POINT (10 10)",
            "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
            2.0,
        );
        compare_dwithin(
            "point in hole, in range of hole edge",
            "POINT (6 10)",
            "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
            2.0,
        );

        compare_dwithin(
            "point in one multipoly",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            0.01,
        );
        compare_dwithin(
            "point out of range of both multipolys",
            "POINT (5 5)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            2.0,
        );
        compare_dwithin(
            "point in range of one multipoly",
            "POINT (3 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            2.0,
        );
    }

    #[test]
    fn multi_point_left() {
        compare_dwithin(
            "one left point in range of point",
            "MULTIPOINT (0 0, 100 100)",
            "POINT (1 0)",
            2.0,
        );
        compare_dwithin(
            "no left point in range of point",
            "MULTIPOINT (0 0, 100 100)",
            "POINT (50 50)",
            2.0,
        );

        compare_dwithin(
            "a left/right point pair in range",
            "MULTIPOINT (0 0, 100 100)",
            "MULTIPOINT (1 0, 101 100)",
            2.0,
        );
        compare_dwithin(
            "no left/right point pair in range",
            "MULTIPOINT (0 0, 100 100)",
            "MULTIPOINT (50 50, 200 200)",
            2.0,
        );

        compare_dwithin(
            "one left point on line",
            "MULTIPOINT (1 0, 100 100)",
            "LINESTRING (0 0, 2 0)",
            0.01,
        );
        compare_dwithin(
            "one left point in range of line",
            "MULTIPOINT (1 1, 100 100)",
            "LINESTRING (0 0, 2 0)",
            2.0,
        );
        compare_dwithin(
            "no left point in range of line",
            "MULTIPOINT (1 5, 100 100)",
            "LINESTRING (0 0, 2 0)",
            2.0,
        );

        compare_dwithin(
            "one left point in poly",
            "MULTIPOINT (2 2, 100 100)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "no left point in range of poly",
            "MULTIPOINT (10 2, 100 100)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        compare_dwithin(
            "one left point in range of poly",
            "MULTIPOINT (5 2, 100 100)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
    }

    #[test]
    fn line_left() {
        compare_dwithin(
            "parallel lines, out of range",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 3, 4 3)",
            2.0,
        );
        compare_dwithin(
            "parallel lines, in range",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 1, 4 1)",
            2.0,
        );
        compare_dwithin(
            "crossing lines",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 -1, 2 1)",
            0.01,
        );
        // Tilt the out-of-range parallel line (y=3) so its right endpoint dips to
        // y=1, one unit from the left line.
        compare_dwithin(
            "tilted line, endpoint in range",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 3, 4 1)",
            2.0,
        );
        // Same tilt, lifted away so the near endpoint is three units off.
        compare_dwithin(
            "tilted line, endpoint out of range",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 5, 4 3)",
            2.0,
        );

        compare_dwithin(
            "line inside poly",
            "LINESTRING (1 1, 3 1)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "line crossing poly edge",
            "LINESTRING (1 1, 5 1)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "line parallel to edge, in range",
            "LINESTRING (0 5, 4 5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        compare_dwithin(
            "line parallel to edge, out of range",
            "LINESTRING (0 7, 4 7)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        // Tilt the out-of-range parallel line (y=7) so its right endpoint dips to
        // y=5, one unit above the top edge.
        compare_dwithin(
            "tilted line vs edge, endpoint in range",
            "LINESTRING (0 7, 4 5)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        // Same tilt, lifted away so the near endpoint is three units above.
        compare_dwithin(
            "tilted line vs edge, endpoint out of range",
            "LINESTRING (0 9, 4 7)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );

        // T-junction pulled apart: the left line's tip sits 1.5 above the middle of
        // the right line (within 2 via the segment interior) but ~5.2 from either of
        // its endpoints. Only true if the shortest (point-to-segment) distance is
        // computed, not endpoint-to-endpoint.
        compare_dwithin(
            "t-junction, nearest point is segment interior",
            "LINESTRING (5 1.5, 5 5)",
            "LINESTRING (0 0, 10 0)",
            2.0,
        );
    }

    #[test]
    fn multi_line_left() {
        compare_dwithin(
            "multilines touching at endpoints",
            "MULTILINESTRING ((0 0, 4 0), (0 10, 4 10))",
            "MULTILINESTRING ((4 0, 8 0), (4 10, 8 10))",
            0.01,
        );

        compare_dwithin(
            "one left segment crosses the line",
            "MULTILINESTRING ((0 0, 4 0), (0 10, 4 10))",
            "LINESTRING (2 -1, 2 1)",
            0.01,
        );
        compare_dwithin(
            "one left segment in range of the line",
            "MULTILINESTRING ((0 0, 4 0), (0 10, 4 10))",
            "LINESTRING (0 1, 4 1)",
            2.0,
        );
        compare_dwithin(
            "no left segment in range of the line",
            "MULTILINESTRING ((0 0, 4 0), (0 10, 4 10))",
            "LINESTRING (0 5, 4 5)",
            2.0,
        );

        compare_dwithin(
            "one left segment crosses the poly",
            "MULTILINESTRING ((2 -1, 2 5), (100 100, 104 100))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            0.01,
        );
        compare_dwithin(
            "one left segment in range of the poly",
            "MULTILINESTRING ((0 5, 4 5), (100 100, 104 100))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
        compare_dwithin(
            "no left segment in range of the poly",
            "MULTILINESTRING ((0 7, 4 7), (100 100, 104 100))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            2.0,
        );
    }

    #[test]
    fn poly_left() {
        compare_dwithin(
            "overlapping polys",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
            0.01,
        );
        compare_dwithin(
            "closest edges in range",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((5 0, 9 0, 9 4, 5 4, 5 0))",
            2.0,
        );
        compare_dwithin(
            "no edge in range",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((7 0, 11 0, 11 4, 7 4, 7 0))",
            2.0,
        );

        compare_dwithin(
            "left poly in hole, out of range of hole edges",
            "POLYGON ((9 9, 11 9, 11 11, 9 11, 9 9))",
            "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
            2.0,
        );
        compare_dwithin(
            "left poly in hole, in range of a hole edge",
            "POLYGON ((6 9, 8 9, 8 11, 6 11, 6 9))",
            "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
            2.0,
        );

        compare_dwithin(
            "polys touching at a vertex",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((4 4, 8 4, 8 8, 4 8, 4 4))",
            2.0,
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_dwithin(
            "a left/right poly pair overlaps",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 0, 12 0, 12 2, 10 2, 10 0)))",
            "MULTIPOLYGON (((1 1, 3 1, 3 3, 1 3, 1 1)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
            0.01,
        );
        compare_dwithin(
            "a left/right poly pair in range",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 0, 12 0, 12 2, 10 2, 10 0)))",
            "MULTIPOLYGON (((3 0, 5 0, 5 2, 3 2, 3 0)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
            2.0,
        );
        compare_dwithin(
            "no left/right poly pair in range",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((10 0, 12 0, 12 2, 10 2, 10 0)))",
            "MULTIPOLYGON (((5 0, 7 0, 7 2, 5 2, 5 0)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
            2.0,
        );

        compare_dwithin(
            "left poly in a right poly's hole, in range of a hole edge",
            "MULTIPOLYGON (((6 9, 8 9, 8 11, 6 11, 6 9)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
            "MULTIPOLYGON (((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5)))",
            2.0,
        );
        compare_dwithin(
            "left poly in a right poly's hole, out of range of hole edges",
            "MULTIPOLYGON (((9 9, 11 9, 11 11, 9 11, 9 9)), ((50 50, 52 50, 52 52, 50 52, 50 50)))",
            "MULTIPOLYGON (((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5)))",
            2.0,
        );
    }
}
