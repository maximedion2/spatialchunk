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

use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

use super::spatial_predicate_udfs::{binary_pair_signature, geo_err, wkb_at};
use crate::geospatial::geos::joinable_geo::{GeoKernel, JoinableGeo, Planar, Segment, Spherical};
use crate::geospatial::geos::raw_geo::{GeoKind, GeoType};
use crate::geospatial::join_predicates::st_intersects::st_intersects;

// *************************************************************
// distance(a, b) -> the minimum distance between a and b, in the crs units for
// geometries, in meters for geographies.
//
// Unlike dwithin there's no threshold, so the natural index can't prune
// anything: it's a brute-force min over every left/right component pair. The
// distance-zero cases (a point inside a poly, a line crossing a poly, two
// overlapping polys, ...) don't surface in the pairwise segment distances, so
// they're caught up front by an intersects check. Once that's ruled out every
// segment pair is disjoint, so `dist_sq_to_disjoint` is valid below.
// *************************************************************
pub(crate) fn st_distance<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> f64 {
    if a.top_bbox_overlaps(b) && st_intersects(a, b) {
        return 0.0;
    }
    K::from_internal_dist_sq(min_dist_sq(a, b))
}

fn min_dist_sq<K: GeoKernel>(a: &JoinableGeo<K>, b: &JoinableGeo<K>) -> f64 {
    match (a, b) {
        (JoinableGeo::Point { points: a_pts, .. }, JoinableGeo::Point { points: b_pts, .. }) => {
            min_point_point::<K>(a_pts, b_pts)
        }

        (JoinableGeo::Point { points, .. }, JoinableGeo::Line { lines, .. }) => {
            min_point_segs::<K, _>(points, lines)
        }
        (JoinableGeo::Point { points, .. }, JoinableGeo::Poly { edges, .. }) => {
            min_point_segs::<K, _>(points, edges)
        }

        (JoinableGeo::Line { lines: l_lines, .. }, JoinableGeo::Line { lines: r_lines, .. }) => {
            min_segs_segs::<K, _, _>(l_lines, r_lines)
        }
        (JoinableGeo::Line { lines, .. }, JoinableGeo::Poly { edges, .. }) => {
            min_segs_segs::<K, _, _>(lines, edges)
        }
        (JoinableGeo::Poly { edges: a_edges, .. }, JoinableGeo::Poly { edges: b_edges, .. }) => {
            min_segs_segs::<K, _, _>(a_edges, b_edges)
        }

        // distance is symmetric; reuse the arms above with the operands swapped.
        (JoinableGeo::Line { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Point { .. })
        | (JoinableGeo::Poly { .. }, JoinableGeo::Line { .. }) => min_dist_sq(b, a),
    }
}

fn min_point_point<K: GeoKernel>(a: &[K::Point], b: &[K::Point]) -> f64 {
    let mut min = f64::INFINITY;
    for p in a {
        for q in b {
            min = min.min(K::dist_sq(p, q));
        }
    }
    min
}

fn min_point_segs<K: GeoKernel, S: Segment<P = K::Point>>(points: &[K::Point], segs: &[S]) -> f64 {
    let mut min = f64::INFINITY;
    for p in points {
        for s in segs {
            min = min.min(K::dist_sq_to_segment(p, s));
        }
    }
    min
}

fn min_segs_segs<K: GeoKernel, L: Segment<P = K::Point>, R: Segment<P = K::Point>>(
    left: &[L],
    right: &[R],
) -> f64 {
    let mut min = f64::INFINITY;
    for l in left {
        for r in right {
            min = min.min(K::dist_sq_to_disjoint(l, r));
        }
    }
    min
}

fn eval_distance<K: GeoKernel>(args: &ScalarFunctionArgs) -> Result<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let left = &arrays[0];
    let right = &arrays[1];
    let n = left.len();

    let mut builder = Float64Builder::with_capacity(n);
    for i in 0..n {
        match (wkb_at(left, i), wkb_at(right, i)) {
            (Some(a), Some(b)) => {
                let a = JoinableGeo::<K>::from_wkb(a).map_err(geo_err)?;
                let b = JoinableGeo::<K>::from_wkb(b).map_err(geo_err)?;
                builder.append_value(st_distance(&a, &b));
            }
            _ => builder.append_null(),
        }
    }
    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

/// `st_distance(geom, geom)` returns the minimum distance between two geos as a
/// `Float64`, row-wise over two WKB columns: in the crs units for geometries, in
/// meters for geographies.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StDistanceUdf {
    signature: Signature,
}

impl Default for StDistanceUdf {
    fn default() -> Self {
        Self {
            signature: binary_pair_signature(),
        }
    }
}

impl ScalarUDFImpl for StDistanceUdf {
    fn name(&self) -> &str {
        "st_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        GeoType::from_fields(
            self.name(),
            args.arg_fields[0].as_ref(),
            args.arg_fields[1].as_ref(),
        )?;
        Ok(Arc::new(Field::new(self.name(), DataType::Float64, true)))
    }

    // both operands have the same type, checked at planning.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match GeoType::from_field(self.name(), args.arg_fields[0].as_ref())?.kind() {
            GeoKind::Geometry => eval_distance::<Planar>(&args),
            GeoKind::Geography => eval_distance::<Spherical>(&args),
        }
    }
}

#[cfg(test)]
mod test_helpers {
    use geos::Geom;

    pub(super) use super::super::super::test_utils::wkt_to_wkb;
    use super::{st_distance, JoinableGeo, Planar};

    // Checks st_distance against GEOS's distance, both operand orders since
    // distance is symmetric.
    pub(super) fn compare_distance(label: &str, wkt_a: &str, wkt_b: &str) {
        let bytes_a = wkt_to_wkb(wkt_a);
        let bytes_b = wkt_to_wkb(wkt_b);

        let geos_a = geos::Geometry::new_from_wkb(&bytes_a).unwrap();
        let geos_b = geos::Geometry::new_from_wkb(&bytes_b).unwrap();
        let geos_result = geos_a.distance(&geos_b).unwrap();

        let joinable_a = JoinableGeo::<Planar>::from_wkb(&bytes_a).unwrap();
        let joinable_b = JoinableGeo::<Planar>::from_wkb(&bytes_b).unwrap();

        assert!(
            (geos_result - st_distance(&joinable_a, &joinable_b)).abs() < 1e-9,
            "{label}"
        );
        assert!(
            (geos_result - st_distance(&joinable_b, &joinable_a)).abs() < 1e-9,
            "{label}, reversed"
        );
    }
}

#[cfg(test)]
mod geography_distance_tests {
    use s2::edgeutil::distance_from_segment;
    use s2::latlng::LatLng;
    use s2::point::Point;

    use super::test_helpers::wkt_to_wkb;
    use super::{st_distance, JoinableGeo, Spherical};
    use crate::geospatial::geos::raw_geo::EARTH_RADIUS_METERS;

    fn geography(wkt: &str) -> JoinableGeo<Spherical> {
        JoinableGeo::from_wkb(&wkt_to_wkb(wkt)).unwrap()
    }

    fn s2_point(lon: f64, lat: f64) -> Point {
        Point::from(LatLng::from_degrees(lat, lon))
    }

    #[test]
    fn intersecting_polys() {
        let a = geography("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))");
        let b = geography("POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))");
        assert_eq!(st_distance(&a, &b), 0.0);
        assert_eq!(st_distance(&b, &a), 0.0);
    }

    // the triangle's apex points at the segment, about 1 degree away, so the
    // distance is the apex's distance to the segment.
    #[test]
    fn triangle_apex_to_segment() {
        let triangle = geography("POLYGON ((1 1, 3 0, 3 2, 1 1))");
        let segment = geography("LINESTRING (0 0, 0 2)");

        let expected = distance_from_segment(
            &s2_point(1.0, 1.0),
            &s2_point(0.0, 0.0),
            &s2_point(0.0, 2.0),
        )
        .rad()
            * EARTH_RADIUS_METERS;
        for distance in [
            st_distance(&triangle, &segment),
            st_distance(&segment, &triangle),
        ] {
            assert!(
                ((distance - expected) / expected).abs() < 1e-6,
                "{distance} vs {expected}"
            );
        }
    }
}

#[cfg(test)]
mod distance_tests {
    use super::test_helpers::compare_distance;

    #[test]
    fn distances() {
        compare_distance("point x point, dist 5", "POINT (0 0)", "POINT (3 4)");
        compare_distance(
            "point x line, nearest is an endpoint",
            "POINT (0 0)",
            "LINESTRING (3 4, 10 4)",
        );
        compare_distance(
            "line x line, nearest is an endpoint pair",
            "LINESTRING (0 0, -3 -4)",
            "LINESTRING (3 4, 6 8)",
        );
        compare_distance(
            "t-junction pulled apart, endpoint to segment interior",
            "LINESTRING (5 6, 5 12)",
            "LINESTRING (0 1, 10 1)",
        );
        compare_distance(
            "t-junction touching, dist 0",
            "LINESTRING (5 0, 5 5)",
            "LINESTRING (0 0, 10 0)",
        );
        compare_distance(
            "lines touching at endpoint, dist 0",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (4 0, 8 0)",
        );
        compare_distance(
            "crossing lines, dist 0",
            "LINESTRING (0 0, 4 0)",
            "LINESTRING (2 -2, 2 2)",
        );
        compare_distance(
            "point in poly, dist 0",
            "POINT (2 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_distance(
            "line crossing poly, dist 0",
            "LINESTRING (2 2, 6 2)",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        );
        compare_distance(
            "intersecting polys, dist 0",
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
        );
        compare_distance(
            "polys, nearest vertex pair at dist 5",
            "POLYGON ((0 0, -2 0, -2 -2, 0 -2, 0 0))",
            "POLYGON ((3 4, 5 4, 5 6, 3 6, 3 4))",
        );
    }
}
