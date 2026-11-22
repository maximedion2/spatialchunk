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

use super::st_within::within_impl;
use crate::geospatial::geos::joinable_geo::{GeoKernel, JoinableGeo};

// *************************************************************
// covered_by(a, b) -> true if every point of a lies in b (interior or boundary).
// Mirrors st_within, but left-interior contact with right's boundary qualifies
// instead of being ignored.
// *************************************************************
pub(crate) fn st_covered_by<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> bool {
    within_impl(a, b, true)
}

// st_covers is forwarded to st_covered_by with swapped arguments.
pub(crate) fn st_covers<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> bool {
    st_covered_by(b, a)
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    use super::{st_covered_by, st_covers};
    use crate::geospatial::geos::joinable_geo::{JoinableGeo, Planar, Spherical};
    pub(super) use crate::geospatial::test_utils::wkt_to_wkb;

    pub(super) fn compare_covered_by(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let mut geos_result = geos_a.covered_by(&geos_b).unwrap();

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();
        let mut joinable_result = st_covered_by(&joinable_a, &joinable_b);

        assert_eq!(geos_result, joinable_result, "{label}");

        geos_result = geos_b.covers(&geos_a).unwrap();
        joinable_result = st_covers(&joinable_b, &joinable_a);

        assert_eq!(geos_result, joinable_result, "{label}, st_covers version");
    }

    // No oracle for geographies, so the expected result is given.
    pub(super) fn check_geography_covered_by(
        label: &str,
        wkt_a: &str,
        wkt_b: &str,
        expected: bool,
    ) {
        let joinable_a = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_a)).unwrap();
        let joinable_b = JoinableGeo::<Spherical>::from_wkb(&wkt_to_wkb(wkt_b)).unwrap();

        assert_eq!(expected, st_covered_by(&joinable_a, &joinable_b), "{label}");
        assert_eq!(
            expected,
            st_covers(&joinable_b, &joinable_a),
            "{label}, st_covers version"
        );
    }
}

#[cfg(test)]
mod geography_covered_by_tests {
    use super::test_helpers::check_geography_covered_by;

    // Points and lines on the boundary, along the equator and meridians, the only
    // great circles whose points are exact in lon/lat. square's bottom edge is
    // along the equator and its sides along meridians, while its top edge from
    // (10 10) to (0 10) bulges north of latitude 10, so a line along latitude 10
    // under it is inside, not on it.
    #[test]
    fn on_boundary() {
        let equator_line = "LINESTRING (0 0, 10 0)";
        let square = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";

        check_geography_covered_by("point on line", "POINT (5 0)", equator_line, true);
        check_geography_covered_by("point on line endpoint", "POINT (0 0)", equator_line, true);
        check_geography_covered_by("point next to line", "POINT (5 0.1)", equator_line, false);

        check_geography_covered_by("point on equator edge", "POINT (5 0)", square, true);
        check_geography_covered_by("point on meridian edge", "POINT (10 5)", square, true);
        check_geography_covered_by(
            "line on equator edge",
            "LINESTRING (2 0, 8 0)",
            square,
            true,
        );
        check_geography_covered_by(
            "line on meridian edge",
            "LINESTRING (10 2, 10 8)",
            square,
            true,
        );
        check_geography_covered_by(
            "line along the equator past the edge",
            "LINESTRING (2 0, 12 0)",
            square,
            false,
        );
        check_geography_covered_by(
            "line along latitude 10, under the top edge",
            "LINESTRING (2 10, 8 10)",
            square,
            true,
        );
    }
}

#[cfg(test)]
mod covered_by_tests {
    use super::test_helpers::compare_covered_by;

    #[test]
    fn point_left() {
        compare_covered_by("point left, case 1", "POINT (1 1)", "POINT (1 1)");
        compare_covered_by("point left, case 2", "POINT (1 1)", "POINT (2 2)");
        compare_covered_by("point left, case 3", "POINT (1 1)", "MULTIPOINT (0 0, 1 1)");
        compare_covered_by("point left, case 4", "POINT (1 1)", "MULTIPOINT (0 0, 2 2)");

        compare_covered_by(
            "point left, case 5",
            "POINT (0.5 0)",
            "LINESTRING (0 0, 1 0)",
        );
        compare_covered_by("point left, case 6", "POINT (2 0)", "LINESTRING (0 0, 1 0)");
        compare_covered_by("point left, case 7", "POINT (0 0)", "LINESTRING (0 0, 1 0)");

        compare_covered_by(
            "point left, case 8",
            "POINT (1 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );
        compare_covered_by(
            "point left, case 9",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 2 0)",
        );

        compare_covered_by(
            "point left, case 10",
            "POINT (0 0)",
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
        );

        compare_covered_by(
            "point left, case 11",
            "POINT (0 1)",
            "LINESTRING (0 3, 0 0, 1 0, 0 1)",
        );

        compare_covered_by(
            "point left, case 12",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2)",
        );

        compare_covered_by(
            "point left, case 13",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, -1 2)",
        );

        compare_covered_by(
            "point left, case 14",
            "POINT (0 2)",
            "LINESTRING (0 4, 0 2, 0 0, 2 0, 2 2, 0 2, -1 2)",
        );

        compare_covered_by(
            "point left, case 15",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 2 2), (0 2, 2 0))",
        );

        compare_covered_by(
            "point left, case 16",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0))",
        );

        compare_covered_by(
            "point left, case 17",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 0), (1 1, 1 2))",
        );

        compare_covered_by(
            "point left, case 18",
            "POINT (1 1)",
            "MULTILINESTRING ((0 0, 1 1, 2 2), (0 2, 1 1, 2 0))",
        );

        compare_covered_by(
            "point left, case 19",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 20",
            "POINT (3 1)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 21",
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 22",
            "POINT (1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 23",
            "POINT (0.5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_covered_by(
            "point left, case 24",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_covered_by(
            "point left, case 25",
            "POINT (1 1)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_covered_by(
            "point left, case 26",
            "POINT (1 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
        );

        compare_covered_by(
            "point left, case 27",
            "POINT (-1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 28",
            "POINT (1 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 29",
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 30",
            "POINT (2 1)",
            "POLYGON ((0 0, 2 0, 2 1, 2 2, 0 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 31",
            "POINT (2 3)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_covered_by(
            "point left, case 32",
            "POINT (2 1)",
            "POLYGON ((0 0, 4 0, 4 4, 2 2, 0 4, 0 0))",
        );

        compare_covered_by(
            "point left, case 33",
            "POINT (1 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_covered_by(
            "point left, case 34",
            "POINT (2.5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))",
        );

        compare_covered_by(
            "point left, case 35",
            "POINT (2 0)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_covered_by(
            "point left, case 36",
            "POINT (-1 4)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );

        compare_covered_by(
            "point left, case 37",
            "POINT (1 2)",
            "POLYGON ((0 0, 0 4, 4 2, 0 0))",
        );
    }

    #[test]
    fn multi_point_left() {
        compare_covered_by(
            "multi point left, case 1",
            "MULTIPOINT (1 1, 2 2)",
            "POINT (1 1)",
        );

        compare_covered_by(
            "multi point left, case 2",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_covered_by(
            "multi point left, case 3",
            "MULTIPOINT (1 1, 2 2, 3 3)",
            "MULTIPOINT (1 1, 2 2)",
        );

        compare_covered_by(
            "multi point left, case 4",
            "MULTIPOINT (1 1, 2 2)",
            "MULTIPOINT (1 1, 2 2, 3 3)",
        );

        compare_covered_by(
            "multi point left, case 5",
            "MULTIPOINT (0 1, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );
        compare_covered_by(
            "multi point left, case 6",
            "MULTIPOINT (1 0, 1 1)",
            "LINESTRING (0 0, 2 0)",
        );
        compare_covered_by(
            "multi point left, case 7",
            "MULTIPOINT (0.5 0, 1.5 0)",
            "LINESTRING (0 0, 2 0)",
        );
        compare_covered_by(
            "multi point left, case 8",
            "MULTIPOINT (1 0, 0 0)",
            "LINESTRING (0 0, 2 0)",
        );
        compare_covered_by(
            "multi point left, case 9",
            "MULTIPOINT (2 0, 1 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );
        compare_covered_by(
            "multi point left, case 10",
            "MULTIPOINT (0 0, 2 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );
        compare_covered_by(
            "multi point left, case 11",
            "MULTIPOINT (1 0, 2 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );
        compare_covered_by(
            "multi point left, case 12",
            "MULTIPOINT (0 0, 3 0)",
            "LINESTRING (0 0, 1 0, 2 0, 3 0)",
        );

        compare_covered_by(
            "multi point left, case 13",
            "MULTIPOINT (2 2, 0 0)",
            "MULTILINESTRING ((0 0, 2 2), (4 0, 2 2), (2 4, 2 2))",
        );

        compare_covered_by(
            "multi point left, case 14",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0))",
        );

        compare_covered_by(
            "multi point left, case 15",
            "MULTIPOINT (2 0, 4 0)",
            "MULTILINESTRING ((0 2, 2 0, 0 -2), (6 2, 4 0, 6 -2), (2 0, 4 0), (2 0, 2 2))",
        );

        compare_covered_by(
            "multi point left, case 16",
            "MULTIPOINT (-1 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 17",
            "MULTIPOINT (2 2, 5 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 18",
            "MULTIPOINT (0 2, 4 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 19",
            "MULTIPOINT (0 2, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 20",
            "MULTIPOINT (0 0, 4 4)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 21",
            "MULTIPOINT (0 0, 2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "multi point left, case 22",
            "MULTIPOINT (3 5, 5 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "multi point left, case 23",
            "MULTIPOINT (3 3, 7 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "multi point left, case 24",
            "MULTIPOINT (5 5, 1 1)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "multi point left, case 25",
            "MULTIPOINT (1 1, 9 9)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "multi point left, case 26",
            "MULTIPOINT (1 3, 1 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "multi point left, case 27",
            "MULTIPOINT (-1 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_covered_by(
            "multi point left, case 28",
            "MULTIPOINT (2 2, 5 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );

        compare_covered_by(
            "multi point left, case 29",
            "MULTIPOINT (2 2, 8 2)",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 0, 10 0, 10 4, 6 4, 6 0)))",
        );
    }

    #[test]
    fn line_left() {
        compare_covered_by("line left, case 1", "LINESTRING (0 0, 4 0)", "POINT (2 0)");
        compare_covered_by(
            "line left, case 2",
            "LINESTRING (0 0, 4 0)",
            "MULTIPOINT (1 0, 3 0)",
        );
        compare_covered_by(
            "line left, case 3",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (0 0, 4 0)",
        );
        compare_covered_by(
            "line left, case 4",
            "LINESTRING (0 0, 4 4)",
            "LINESTRING (0 4, 4 0)",
        );
        compare_covered_by(
            "line left, case 5",
            "LINESTRING (0 0, 2 2, 4 0)",
            "LINESTRING (0 4, 2 2, 4 4)",
        );
        compare_covered_by(
            "line left, case 6",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (5 0, 7 0)",
        );
        compare_covered_by(
            "line left, case 7",
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (3 0, 5 0)",
        );
        compare_covered_by(
            "line left, case 8",
            "LINESTRING (0 0, 3 0)",
            "LINESTRING (2 0, 5 0)",
        );
        compare_covered_by(
            "line left, case 9",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 4 0)",
        );
        compare_covered_by(
            "line left, case 10",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (1 0, 3 0)",
        );
        compare_covered_by(
            "line left, case 11",
            "LINESTRING (0 0, 4 0, 2 2)",
            "LINESTRING (0 0, 4 0, 4 4)",
        );

        compare_covered_by(
            "line left, case 12",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
            "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)",
        );

        compare_covered_by(
            "line left, case 13",
            "LINESTRING (0 2, 4 2)",
            "LINESTRING (0 0, 2 2, 4 0)",
        );

        compare_covered_by(
            "line left, case 14",
            "LINESTRING (0 0, 2 0, 2 2, 4 2)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 2 2), (2 2, 4 2))",
        );

        compare_covered_by(
            "line left, case 15",
            "LINESTRING (1 0, 3 0)",
            "LINESTRING (0 0, 2 0, 4 0)",
        );

        compare_covered_by(
            "line left, case 16",
            "LINESTRING (1 0, 3 0)",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
        );

        compare_covered_by(
            "line left, case 17",
            "LINESTRING (0 0, 4 0)",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_covered_by(
            "line left, case 18",
            "LINESTRING (0 0, 2 0, 2 2)",
            "MULTILINESTRING ((-1 0, 3 0), (2 -1, 2 3))",
        );

        compare_covered_by(
            "line left, case 19",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0), (4 0, 6 0))",
        );

        compare_covered_by(
            "line left, case 20",
            "LINESTRING (1 0, 5 0)",
            "LINESTRING (0 0, 4 0, 4 2, 5 2, 5 0, 2 0)",
        );

        compare_covered_by(
            "line left, case 21",
            "LINESTRING (0 0, 6 0)",
            "MULTILINESTRING ((0 0, 3 0), (2 0, 5 0))",
        );

        compare_covered_by(
            "line left, case 22",
            "LINESTRING (-2 3, -1 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 23",
            "LINESTRING (1 3, 5 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 24",
            "LINESTRING (-1 3, 3 3)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 25",
            "LINESTRING (-2 -2, 2 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 26",
            "LINESTRING (-1 0, 3 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 27",
            "LINESTRING (1 0, 5 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 28",
            "LINESTRING (0 0, 6 0, 6 6, 0 6, 0 0)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 29",
            "LINESTRING (1 5, 2 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "line left, case 30",
            "LINESTRING (4 5, 6 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "line left, case 31",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "line left, case 32",
            "LINESTRING (1 1, 5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "line left, case 33",
            "LINESTRING (3 3, 3 7)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "line left, case 34",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)))",
        );

        compare_covered_by(
            "line left, case 35",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 4 4, 2 2)), ((2 2, 1 4, 3 4, 2 2)))",
        );

        compare_covered_by(
            "line left, case 36",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 2 2, 0 4, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_covered_by(
            "line left, case 37",
            "LINESTRING (0.5 2, 3.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_covered_by(
            "line left, case 38",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 39",
            "LINESTRING (-1 6, 4 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 40",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 41",
            "LINESTRING (-1 5, 9 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 42",
            "LINESTRING (1 5, 7 5)",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );

        compare_covered_by(
            "line left, case 43",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 44",
            "LINESTRING (1 5, 5 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 45",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 46",
            "LINESTRING (0 -2, 0 2)",
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))",
        );

        compare_covered_by(
            "line left, case 47",
            "LINESTRING (-1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "line left, case 48",
            "LINESTRING (1 4, 6 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "line left, case 49",
            "LINESTRING (6 4, -1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "line left, case 50",
            "LINESTRING (6 4, 1 4)",
            "POLYGON ((0 0, 8 0, 8 8, 4 8, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "line left, case 51",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 2, 4 1, 3 0, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_covered_by(
            "line left, case 52",
            "LINESTRING (0.5 2, 4 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 2, 4 2, 4 4, 2 4, 2 2)))",
        );

        compare_covered_by(
            "line left, case 53",
            "LINESTRING (2 1, 4 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)), ((4 0, 6 0, 6 2, 4 2, 4 0)))",
        );

        compare_covered_by(
            "line left, case 54",
            "LINESTRING (0.5 2, 7.5 2)",
            "MULTIPOLYGON (((0 0, 0 4, 2 2, 0 0)), ((2 2, 4 0, 6 2, 4 4, 2 2)), ((6 2, 8 0, 8 4, 6 2)))",
        );

        compare_covered_by(
            "line left, case 55",
            "LINESTRING (1 5, 7 5)",
            "POLYGON ((0 0, 0 8, 4 5, 8 8, 8 0, 0 0))",
        );

        compare_covered_by(
            "line left, case 56",
            "LINESTRING (5 5, 1 5)",
            "POLYGON ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 57",
            "LINESTRING (1 6, 4 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_covered_by(
            "line left, case 58",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 3 6, 3 8, 0 8, 0 0))",
        );

        compare_covered_by(
            "line left, case 59",
            "LINESTRING (4 6, 1 6)",
            "POLYGON ((0 0, 0 8, 3 8, 3 6, 5 6, 5 8, 8 8, 8 0, 0 0))",
        );

        compare_covered_by(
            "line left, case 60",
            "LINESTRING (1 4, 7 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "line left, case 61",
            "LINESTRING (0 4, 5.5 4)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "line left, case 62",
            "LINESTRING (1 2, 7 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "line left, case 63",
            "LINESTRING (0 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 3 0, 3 4, 0 4, 0 0)), ((5 0, 8 0, 8 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "line left, case 64",
            "LINESTRING (0 2, 8 2)",
            "POLYGON ((4 0, 8 0, 8 4, 4 4, 4 0))",
        );

        compare_covered_by(
            "line left, case 65",
            "LINESTRING (0.5 2, 5.5 2)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 2, 3 1, 4 2, 3 3, 2 2)))",
        );

        compare_covered_by(
            "line left, case 66",
            "LINESTRING (1 1, 5 1)",
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)), ((2 1, 3 0, 4 1, 3 2, 2 1)))",
        );

        compare_covered_by(
            "line left, case 67",
            "LINESTRING (1 6, 7 6)",
            "POLYGON ((0 0, 8 0, 8 8, 5 8, 5 6, 4 6, 3 6, 3 8, 0 8, 0 0))",
        );
    }

    #[test]
    fn multi_line_left() {
        compare_covered_by(
            "multi line left, case 1",
            "MULTILINESTRING ((1 0, 3 0), (1 2, 3 2))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_covered_by(
            "multi line left, case 2",
            "MULTILINESTRING ((1 0, 3 0), (1 1, 3 1))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_covered_by(
            "multi line left, case 3",
            "MULTILINESTRING ((1 1, 3 1), (1 3, 3 3))",
            "MULTILINESTRING ((0 0, 4 0), (0 2, 4 2))",
        );

        compare_covered_by(
            "multi line left, case 4",
            "MULTILINESTRING ((1 1, 3 1), (6 1, 8 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "multi line left, case 5",
            "MULTILINESTRING ((1 1, 3 1), (10 1, 12 1))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "multi line left, case 6",
            "MULTILINESTRING ((10 1, 12 1), (10 3, 12 3))",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 0, 9 0, 9 4, 5 4, 5 0)))",
        );

        compare_covered_by(
            "multi line left, case 7",
            "MULTILINESTRING ((0 0, 2 0), (2 0, 4 0))",
            "LINESTRING (0 0, 2 0, 4 0)",
        );
    }

    #[test]
    fn poly_left() {
        compare_covered_by(
            "poly left, case 1",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POINT (2 2)",
        );
        compare_covered_by(
            "poly left, case 2",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "LINESTRING (0 0, 4 4)",
        );

        compare_covered_by(
            "poly left, case 3",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "poly left, case 4",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "poly left, case 5",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "poly left, case 6",
            "POLYGON ((5 5, 9 5, 9 9, 5 9, 5 5))",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );

        compare_covered_by(
            "poly left, case 7",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "poly left, case 8",
            "POLYGON ((1 1, 9 1, 9 9, 1 9, 1 1), (2 2, 2 8, 8 8, 8 2, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "poly left, case 9",
            "POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "poly left, case 10",
            "POLYGON ((2 2, 5 2, 5 5, 2 5, 2 2))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
        );

        compare_covered_by(
            "poly left, case 11",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((0 2, 4 2, 4 6, 0 6, 0 2))",
        );

        compare_covered_by(
            "poly left, case 12",
            "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))",
            "MULTIPOLYGON (((0 2, 2 2, 2 4, 0 4, 0 2)), ((2 4, 4 4, 4 6, 2 6, 2 4)), ((4 2, 6 2, 6 4, 4 4, 4 2)), ((2 0, 4 0, 4 2, 2 2, 2 0)))",
        );

        compare_covered_by(
            "poly left, case 13",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "MULTIPOLYGON (
                ((1 1, 2 0.5, 3 1, 2 1.5, 1 1)),
                ((3 1, 3.5 2, 3 3, 2.5 2, 3 1)),
                ((3 3, 2 3.5, 1 3, 2 2.5, 3 3)),
                ((1 3, 0.5 2, 1 1, 1.5 2, 1 3)))
            ",
        );

        compare_covered_by(
            "poly left, case 14",
            "POLYGON ((1 4, 7 4, 7 5, 1 5, 1 4))",
            "MULTIPOLYGON (((4 5, 3 7, 5 7, 4 5)), ((0 0, 8 0, 8 8, 4 5, 0 8, 0 0)))",
        );
    }

    #[test]
    fn multi_poly_left() {
        compare_covered_by(
            "multi poly left, case 1",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((3 1, 4 1, 4 2, 3 2, 3 1)))",
            "POLYGON ((0 0, 5 0, 5 3, 0 3, 0 0))",
        );

        compare_covered_by(
            "multi poly left, case 2",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_covered_by(
            "multi poly left, case 3",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)), ((10 0, 14 0, 14 3, 10 3, 10 0)))",
        );

        compare_covered_by(
            "multi poly left, case 4",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((6 1, 7 1, 7 2, 6 2, 6 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );

        compare_covered_by(
            "multi poly left, case 5",
            "MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((10 1, 11 1, 11 2, 10 2, 10 1)))",
            "MULTIPOLYGON (((0 0, 4 0, 4 3, 0 3, 0 0)), ((5 0, 9 0, 9 3, 5 3, 5 0)))",
        );

        compare_covered_by(
            "multi poly left, case 6",
            "MULTIPOLYGON (((6 1, 7 1, 7 2, 6 2, 6 1)), ((8 1, 9 1, 9 2, 8 2, 8 1)))",
            "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        );
    }

    #[test]
    fn empty_geo() {
        let sq = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        compare_covered_by("empty multipoint left", "MULTIPOINT EMPTY", sq);
        compare_covered_by("empty multilinestring left", "MULTILINESTRING EMPTY", sq);
        compare_covered_by("empty polygon left", "POLYGON EMPTY", sq);
        compare_covered_by("empty multipoint right", sq, "MULTIPOINT EMPTY");
        compare_covered_by("empty multilinestring right", sq, "MULTILINESTRING EMPTY");
        compare_covered_by("empty polygon right", sq, "POLYGON EMPTY");
    }
}
