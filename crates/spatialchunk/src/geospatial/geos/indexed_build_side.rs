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

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

use arrow::compute::{interleave, nullif};
use arrow_array::{
    new_null_array, Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, RecordBatch,
};
use arrow_schema::DataType;
use datafusion::common::JoinSide;
use datafusion::error::{DataFusionError, Result};
use futures::TryStreamExt;
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{Node, RTree as GeoRTree, RTreeBuilder, RTreeIndex};
use geo_types::Rect;

use super::boxed_geo_batch::{BBoxedGeoBatch, BBoxedGeoStream};
use super::joinable_geo::{GeoError, GeoKernel, JoinableGeo, Planar, Spherical};
use crate::geospatial::join_predicates::st_coveredby::{st_covered_by, st_covers};
use crate::geospatial::join_predicates::st_dwithin::st_dwithin;
use crate::geospatial::join_predicates::st_intersects::st_intersects;
use crate::geospatial::join_predicates::st_touches::st_touches;
use crate::geospatial::join_predicates::st_within::{st_contains, st_within};
use crate::geospatial::join_predicates::SpatialRelationType;

// In-place partition of the first boundary probe entries: keep those whose
// rect overlaps the given node bbox by swapping survivors to the front; returns
// the survivor count.
fn partition_bbox(
    rects: &mut [Rect<f32>],
    ids: &mut [usize],
    boundary: usize,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
) -> usize {
    let mut w = 0;
    for r in 0..boundary {
        let rect = &rects[r];
        if rect.min().x <= max_x
            && rect.max().x >= min_x
            && rect.min().y <= max_y
            && rect.max().y >= min_y
        {
            if w != r {
                rects.swap(w, r);
                ids.swap(w, r);
            }
            w += 1;
        }
    }
    w
}

// Vectorized multi-probe DFS over a geo-index R-tree node. Partitions the live
// probe prefix [0..boundary) against node's bbox; on a leaf (one geometry) emits
// surviving probes paired with the leaf's insertion_index (the build geo-id);
// on a parent recurses into each child with the narrowed boundary.
fn descend<T: RTreeIndex<f32>>(
    node: Node<'_, f32, T>,
    rects: &mut [Rect<f32>],
    ids: &mut [usize],
    boundary: usize,
    probe_out: &mut Vec<usize>,
    build_out: &mut Vec<usize>,
) {
    let new_boundary = partition_bbox(
        rects,
        ids,
        boundary,
        node.min_x(),
        node.min_y(),
        node.max_x(),
        node.max_y(),
    );
    if new_boundary == 0 {
        return;
    }

    if node.is_leaf() {
        let geo_id = node.insertion_index_unchecked() as usize;
        for &probe_id in &ids[..new_boundary] {
            probe_out.push(probe_id);
            build_out.push(geo_id);
        }
    } else if let Some(children) = node.children() {
        for child in children {
            descend(child, rects, ids, new_boundary, probe_out, build_out);
        }
    }
}

fn wkb_at(array: &ArrayRef, row_idx: usize) -> &[u8] {
    match array.data_type() {
        DataType::Binary => array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("geometry array is Binary")
            .value(row_idx),
        DataType::BinaryView => array
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .expect("geometry array is BinaryView")
            .value(row_idx),
        other => panic!("unsupported geometry array type: {other}"),
    }
}

// Lazy, thread-safe cache of parsed build-side JoinableGeo's.
pub(crate) struct CachedJoinableGeo<K: GeoKernel> {
    geo_arrays: Vec<ArrayRef>,
    cache: Vec<Vec<AtomicPtr<JoinableGeo<K>>>>,
}

impl<K: GeoKernel> CachedJoinableGeo<K> {
    pub(crate) fn new(geo_arrays: Vec<ArrayRef>) -> Self {
        let cache = geo_arrays
            .iter()
            .map(|arr| {
                (0..arr.len())
                    .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                    .collect()
            })
            .collect();
        Self { geo_arrays, cache }
    }

    pub(crate) fn get_or_parse(
        &self,
        positions: &[(usize, usize)],
    ) -> Result<Vec<&JoinableGeo<K>>, GeoError> {
        positions
            .iter()
            .map(|&(batch_idx, row_idx)| {
                let slot = &self.cache[batch_idx][row_idx];
                let ptr = slot.load(Ordering::Acquire);
                if !ptr.is_null() {
                    return Ok(unsafe { &*ptr });
                }
                let arr = &self.geo_arrays[batch_idx];
                let bytes: &[u8] = wkb_at(arr, row_idx);
                let new_ptr = Box::into_raw(Box::new(JoinableGeo::from_wkb(bytes)?));
                match slot.compare_exchange(
                    std::ptr::null_mut(),
                    new_ptr,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => Ok(unsafe { &*new_ptr }),
                    Err(existing) => {
                        unsafe { drop(Box::from_raw(new_ptr)) };
                        Ok(unsafe { &*existing })
                    }
                }
            })
            .collect()
    }
}

impl<K: GeoKernel> Drop for CachedJoinableGeo<K> {
    fn drop(&mut self) {
        for batch in &self.cache {
            for slot in batch {
                let p = slot.load(Ordering::Acquire);
                if !p.is_null() {
                    unsafe { drop(Box::from_raw(p)) };
                }
            }
        }
    }
}

// Evaluates a spatial relation between a matched (build, probe) geometry pair.
// side is which side the build geometry occupies.
fn evaluate_relation<K: GeoKernel>(
    predicate: &SpatialRelationType,
    build: &JoinableGeo<K>,
    probe: &JoinableGeo<K>,
    side: JoinSide,
    touches: fn(&JoinableGeo<K>, &JoinableGeo<K>) -> bool,
) -> Result<bool, GeoError> {
    let (left, right) = match side {
        JoinSide::Left => (build, probe),
        JoinSide::Right => (probe, build),
        JoinSide::None => return Err(GeoError::InvalidJoinSide),
    };
    Ok(match predicate {
        SpatialRelationType::Within => st_within(left, right),
        SpatialRelationType::Contains => st_contains(left, right),
        SpatialRelationType::CoveredBy => st_covered_by(left, right),
        SpatialRelationType::Covers => st_covers(left, right),
        SpatialRelationType::Intersects => st_intersects(left, right),
        SpatialRelationType::Touches => touches(left, right),
        SpatialRelationType::DWithin { distance } => st_dwithin(left, right, *distance),
    })
}

enum BuildGeos {
    Planar(CachedJoinableGeo<Planar>),
    Spherical(CachedJoinableGeo<Spherical>),
}

pub(crate) struct IndexedBuildSide {
    tree: GeoRTree<f32>,
    joinable_geos: BuildGeos,
    batch_positions: Vec<(usize, usize)>,
    batches: Vec<RecordBatch>,
    visited: Option<Vec<Vec<AtomicBool>>>,
    remaining_probes: AtomicUsize,
}

impl IndexedBuildSide {
    // Vectorized bbox traversal of the geo-index R-tree: returns parallel
    // (probe_ids, build_geo_ids) candidate pairs whose bboxes overlap.
    pub(crate) fn traverse(
        &self,
        mut probe_rects: Vec<Rect<f32>>,
        mut probe_ids: Vec<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let boundary = probe_ids.len();
        if boundary == 0 || self.tree.num_items() == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut probe_out = Vec::new();
        let mut build_out = Vec::new();
        descend(
            self.tree.root(),
            &mut probe_rects,
            &mut probe_ids,
            boundary,
            &mut probe_out,
            &mut build_out,
        );
        (probe_out, build_out)
    }

    pub(crate) fn traverse_with_refinement(
        &self,
        probe_rects: Vec<Rect<f32>>,
        probe_ids: Vec<usize>,
        probe_geo_array: &ArrayRef,
        predicate: &SpatialRelationType,
        side: JoinSide,
    ) -> Result<(Vec<usize>, Vec<usize>), GeoError> {
        // First find build side - probe side pairs to check
        let (probe_matched, build_matched) = self.traverse(probe_rects, probe_ids);

        match &self.joinable_geos {
            BuildGeos::Planar(cache) => self.refine(
                cache,
                probe_matched,
                build_matched,
                probe_geo_array,
                predicate,
                side,
                st_touches,
            ),
            BuildGeos::Spherical(cache) => self.refine(
                cache,
                probe_matched,
                build_matched,
                probe_geo_array,
                predicate,
                side,
                |_, _| unreachable!("st_touches on geographies is rejected at planning"),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine<K: GeoKernel>(
        &self,
        cache: &CachedJoinableGeo<K>,
        probe_matched: Vec<usize>,
        build_matched: Vec<usize>,
        probe_geo_array: &ArrayRef,
        predicate: &SpatialRelationType,
        side: JoinSide,
        touches: fn(&JoinableGeo<K>, &JoinableGeo<K>) -> bool,
    ) -> Result<(Vec<usize>, Vec<usize>), GeoError> {
        // Construct the geometries from teh binaty arrays (if they
        // they have not already been parsed, that's the lazy part).
        let build_positions: Vec<(usize, usize)> = build_matched
            .iter()
            .map(|&geo_id| self.batch_positions[geo_id])
            .collect();
        let build_geos = cache.get_or_parse(&build_positions)?;
        let mut probe_geos: HashMap<usize, JoinableGeo<K>> = HashMap::new();
        for &probe_id in &probe_matched {
            if let Entry::Vacant(e) = probe_geos.entry(probe_id) {
                let geo = JoinableGeo::from_wkb(wkb_at(probe_geo_array, probe_id))?;
                e.insert(geo);
            }
        }

        // This, check each matched pairs to see if they actually pass the
        // predicate.
        let mut probe_out = Vec::new();
        let mut build_out = Vec::new();

        for ((&probe_id, &geo_id), build_geo) in probe_matched
            .iter()
            .zip(build_matched.iter())
            .zip(build_geos.iter().copied())
        {
            let probe_geo = &probe_geos[&probe_id];
            if evaluate_relation(predicate, build_geo, probe_geo, side, touches)? {
                probe_out.push(probe_id);
                build_out.push(geo_id);
            }
        }

        Ok((probe_out, build_out))
    }

    fn resolve(&self, geo_id: usize) -> (usize, usize) {
        self.batch_positions[geo_id]
    }

    // Gathers `col_idx` from the build batches for the given geo-ids. None entries
    // become null rows (used for unmatched-probe rows in outer joins).
    pub(crate) fn interleave_column_opt(
        &self,
        geo_ids: &[Option<usize>],
        col_idx: usize,
    ) -> Result<ArrayRef> {
        let arrays: Vec<&dyn Array> = self
            .batches
            .iter()
            .map(|b| b.column(col_idx).as_ref())
            .collect();

        // All-null shortcut.
        if geo_ids.iter().all(|id| id.is_none()) {
            let data_type = arrays[0].data_type().clone();
            return Ok(new_null_array(&data_type, geo_ids.len()));
        }

        // None entries use a (0,0) placeholder position and are nulled out afterwards.
        let indices: Vec<(usize, usize)> = geo_ids
            .iter()
            .map(|id| id.map(|id| self.resolve(id)).unwrap_or((0, 0)))
            .collect();
        let array = interleave(&arrays, &indices)?;

        if geo_ids.iter().any(|id| id.is_none()) {
            let mask =
                BooleanArray::from(geo_ids.iter().map(|id| id.is_none()).collect::<Vec<_>>());
            Ok(nullif(&array, &mask)?)
        } else {
            Ok(array)
        }
    }

    pub(crate) fn mark_visited(&self, geo_ids: &[usize]) {
        if let Some(visited) = &self.visited {
            for &geo_id in geo_ids {
                let (batch_idx, row_idx) = self.resolve(geo_id);
                visited[batch_idx][row_idx].store(true, Ordering::Release);
            }
        }
    }

    pub(crate) fn unvisited_positions(&self) -> Option<Vec<usize>> {
        self.visited.as_ref().map(|visited| {
            self.batch_positions
                .iter()
                .enumerate()
                .filter(|(_, &(batch_idx, row_idx))| {
                    !visited[batch_idx][row_idx].load(Ordering::Acquire)
                })
                .map(|(geo_id, _)| geo_id)
                .collect()
        })
    }

    pub(crate) fn decrement_remaining_probes(&self) -> usize {
        self.remaining_probes.fetch_sub(1, Ordering::SeqCst)
    }
}

// Indexed build side builder, that accumulates batches of data that contain
// a geo column, and builds all the machinery to be able to join that data
// against the probe side. The data model here is that a partition can
// contain multiple batches.
struct IndexedBuildSideBuilder {
    batches: Vec<RecordBatch>,
    geo_arrays: Vec<ArrayRef>,
    batch_positions: Vec<(usize, usize)>,
    rects: Vec<Rect<f32>>,
}

impl IndexedBuildSideBuilder {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
            geo_arrays: Vec::new(),
            batch_positions: Vec::new(),
            rects: Vec::new(),
        }
    }

    fn add_partition(&mut self, batches: Vec<BBoxedGeoBatch>) {
        for batch in batches {
            self.add_batch(batch);
        }
    }

    fn add_batch(&mut self, boxed: BBoxedGeoBatch) {
        let batch_idx = self.batches.len();
        self.batch_positions
            .extend(boxed.geo_ids.iter().map(|&row_idx| (batch_idx, row_idx)));
        self.rects.extend(boxed.rects);
        self.batches.push(boxed.batch);
        self.geo_arrays.push(boxed.geo_array);
    }

    fn build(
        self,
        needs_visited: bool,
        num_probe_partitions: usize,
        is_geography: bool,
    ) -> IndexedBuildSide {
        // Build the geo-index R-tree. add returns the insertion index in call
        // order, so adding rects in geo-id order makes insertion_index == geo_id
        // during traversal.
        let mut builder = RTreeBuilder::<f32>::new(self.rects.len() as u32);
        for rect in &self.rects {
            builder.add(rect.min().x, rect.min().y, rect.max().x, rect.max().y);
        }
        let tree = builder.finish::<HilbertSort>();

        let visited = needs_visited.then(|| {
            self.batches
                .iter()
                .map(|b| (0..b.num_rows()).map(|_| AtomicBool::new(false)).collect())
                .collect()
        });

        let joinable_geos = if is_geography {
            BuildGeos::Spherical(CachedJoinableGeo::new(self.geo_arrays.clone()))
        } else {
            BuildGeos::Planar(CachedJoinableGeo::new(self.geo_arrays.clone()))
        };

        IndexedBuildSide {
            joinable_geos,
            tree,
            batch_positions: self.batch_positions,
            batches: self.batches,
            visited,
            remaining_probes: AtomicUsize::new(num_probe_partitions),
        }
    }
}

pub(crate) async fn build_from_streams(
    streams: Vec<BBoxedGeoStream>,
    needs_visited: bool,
    num_probe_partitions: usize,
    is_geography: bool,
) -> Result<IndexedBuildSide> {
    let mut join_set = tokio::task::JoinSet::new();

    for stream in streams {
        join_set.spawn(async move {
            let batches: Vec<BBoxedGeoBatch> = stream.try_collect().await?;
            Ok::<_, DataFusionError>(batches)
        });
    }

    let mut builder = IndexedBuildSideBuilder::new();
    while let Some(res) = join_set.join_next().await {
        let batches = res.map_err(|e| DataFusionError::Internal(e.to_string()))??;
        builder.add_partition(batches);
    }

    Ok(builder.build(needs_visited, num_probe_partitions, is_geography))
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::sync::Arc;

    use arrow_array::{BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::PhysicalExpr;

    use super::{IndexedBuildSide, IndexedBuildSideBuilder};
    use crate::geospatial::geos::boxed_geo_batch::{bboxed_geo_batch, BBoxedGeoBatch};
    use crate::geospatial::geos::raw_geo::GeoType;
    use crate::geospatial::test_utils::wkt_to_wkb;
    use crate::geospatial::udfs::udf_helpers::{geometry_field, GeoMetadata};

    pub(crate) fn make_geo_batch(
        wkbs: &[Vec<u8>],
        geo_col: &str,
        value_col: &str,
    ) -> BBoxedGeoBatch {
        let schema = Arc::new(Schema::new(vec![
            geometry_field(geo_col, DataType::Binary, false, &GeoMetadata::default())
                .as_ref()
                .clone(),
            Field::new(value_col, DataType::Int32, false),
        ]));
        let values: Vec<i32> = (0..wkbs.len() as i32).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|b| b.as_slice()),
                )),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();
        let geo_expr = Arc::new(Column::new(geo_col, 0)) as Arc<dyn PhysicalExpr>;
        bboxed_geo_batch(batch, &geo_expr, None).unwrap()
    }

    fn make_build_batch(
        wkts: &[&str],
        geo_col: &str,
        value_col: &str,
        row_offset: i32,
    ) -> BBoxedGeoBatch {
        let schema = Arc::new(Schema::new(vec![
            geometry_field(geo_col, DataType::Binary, false, &GeoMetadata::default())
                .as_ref()
                .clone(),
            Field::new(value_col, DataType::Int32, false),
        ]));
        let wkbs: Vec<Vec<u8>> = wkts.iter().map(|wkt| wkt_to_wkb(wkt)).collect();
        let values: Vec<i32> = (row_offset..row_offset + wkts.len() as i32).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|b| b.as_slice()),
                )),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();
        let geo_expr = Arc::new(Column::new(geo_col, 0)) as Arc<dyn PhysicalExpr>;
        bboxed_geo_batch(batch, &geo_expr, None).unwrap()
    }

    pub(crate) fn make_indexed_build_side(
        wkts: &[&str],
        num_build_partitions: usize,
        num_probe_partitions: usize,
        needs_visited: bool,
        geo_col: &str,
        value_col: &str,
    ) -> Arc<IndexedBuildSide> {
        let chunk_size = wkts.len().div_ceil(num_build_partitions);
        let mut builder = IndexedBuildSideBuilder::new();
        let mut row_offset: i32 = 0;
        for chunk in wkts.chunks(chunk_size) {
            builder.add_partition(vec![make_build_batch(
                chunk, geo_col, value_col, row_offset,
            )]);
            row_offset += chunk.len() as i32;
        }
        Arc::new(builder.build(needs_visited, num_probe_partitions, false))
    }

    // A single partition index of `wkts`, with the type `geo_type`, e.g. a
    // geography, which gets geodesic bboxes.
    pub(crate) fn make_typed_indexed_build_side(
        wkts: &[&str],
        geo_type: &GeoType,
    ) -> Arc<IndexedBuildSide> {
        let schema = Arc::new(Schema::new(vec![geo_type
            .to_field("geometry", DataType::Binary, false)
            .as_ref()
            .clone()]));
        let wkbs: Vec<Vec<u8>> = wkts.iter().map(|wkt| wkt_to_wkb(wkt)).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from_iter_values(
                wkbs.iter().map(|b| b.as_slice()),
            ))],
        )
        .unwrap();
        let geo_expr = Arc::new(Column::new("geometry", 0)) as Arc<dyn PhysicalExpr>;
        let mut builder = IndexedBuildSideBuilder::new();
        builder.add_partition(vec![bboxed_geo_batch(batch, &geo_expr, None).unwrap()]);
        Arc::new(builder.build(false, 1, geo_type.is_geography()))
    }
}

#[cfg(test)]
mod indexed_build_side_tests {
    use datafusion::common::JoinSide;

    use super::test_helpers::{make_geo_batch, make_indexed_build_side};
    use super::*;
    use crate::geospatial::join_predicates::SpatialRelationType;
    use crate::geospatial::test_utils::make_squares;

    fn traverse_within_and_mark(
        index: &IndexedBuildSide,
        probe_boxed: BBoxedGeoBatch,
        expected_pairs: &[(usize, usize)],
    ) {
        let (probe_ids, build_ids) = index
            .traverse_with_refinement(
                probe_boxed.rects,
                probe_boxed.geo_ids,
                &probe_boxed.geo_array,
                &SpatialRelationType::Within,
                JoinSide::Right,
            )
            .unwrap();
        let mut pairs: Vec<(usize, usize)> = probe_ids
            .iter()
            .copied()
            .zip(build_ids.iter().copied())
            .collect();
        pairs.sort();
        assert_eq!(pairs, expected_pairs);
        index.mark_visited(&build_ids);
    }

    #[test]
    fn test_traverse_with_refinement_and_visited() {
        // Build side: 4 large squares, 2 per partition.
        // geo_id 0 = partition 0 row 0 → [0,10]×[0,10]
        // geo_id 1 = partition 0 row 1 → [5,15]×[0,10]  (overlaps geo_id 0)
        // geo_id 2 = partition 1 row 0 → [20,30]×[0,10]
        // geo_id 3 = partition 1 row 1 → [40,50]×[0,10]  (unmatched)
        let index = make_indexed_build_side(
            &[
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
                "POLYGON((5 0, 15 0, 15 10, 5 10, 5 0))",
                "POLYGON((20 0, 30 0, 30 10, 20 10, 20 0))",
                "POLYGON((40 0, 50 0, 50 10, 40 10, 40 0))",
            ],
            2,
            1,
            true,
            "geometry",
            "value",
        );

        // Probe side: 3 unit squares.
        // probe 0 → [6,7]×[1,2]:   within geo_id 0 and 1
        // probe 1 → [21,22]×[1,2]: within geo_id 2 only
        // probe 2 → [60,61]×[0,1]: no match
        let probe_wkbs = make_squares(&[Some((6.0, 1.0)), Some((21.0, 1.0)), Some((60.0, 0.0))]);
        let probe_boxed = make_geo_batch(&probe_wkbs, "geometry", "value");
        traverse_within_and_mark(&index, probe_boxed, &[(0, 0), (0, 1), (1, 2)]);

        let mut unvisited = index.unvisited_positions().unwrap();
        unvisited.sort();
        assert_eq!(unvisited, vec![3]);
    }

    #[test]
    fn test_traverse_empty_probe_geometry() {
        // Same build side as the base test (geo_ids 0-3, geo_id 3 unmatched).
        let index = make_indexed_build_side(
            &[
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
                "POLYGON((5 0, 15 0, 15 10, 5 10, 5 0))",
                "POLYGON((20 0, 30 0, 30 10, 20 10, 20 0))",
                "POLYGON((40 0, 50 0, 50 10, 40 10, 40 0))",
            ],
            2,
            1,
            true,
            "geometry",
            "value",
        );

        // Probe row 2 is an empty polygon — BBoxedGeoBatch skips it, so only
        // probe rows 0 and 1 participate in the traversal.
        let probe_wkbs = make_squares(&[Some((6.0, 1.0)), Some((21.0, 1.0)), None]);
        let probe_boxed = make_geo_batch(&probe_wkbs, "geometry", "value");
        assert_eq!(probe_boxed.geo_ids, vec![0, 1]);
        traverse_within_and_mark(&index, probe_boxed, &[(0, 0), (0, 1), (1, 2)]);

        let mut unvisited = index.unvisited_positions().unwrap();
        unvisited.sort();
        assert_eq!(unvisited, vec![3]);
    }

    #[test]
    fn test_traverse_empty_build_geometry() {
        // Partition 1 row 1 is an empty polygon — BBoxedGeoBatch skips it, so
        // it never gets a geo_id. Only geo_ids 0, 1, 2 exist.
        let index = make_indexed_build_side(
            &[
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
                "POLYGON((5 0, 15 0, 15 10, 5 10, 5 0))",
                "POLYGON((20 0, 30 0, 30 10, 20 10, 20 0))",
                "POLYGON EMPTY",
            ],
            2,
            1,
            true,
            "geometry",
            "value",
        );

        // Same probe side as the base test.
        let probe_wkbs = make_squares(&[Some((6.0, 1.0)), Some((21.0, 1.0)), Some((60.0, 0.0))]);
        let probe_boxed = make_geo_batch(&probe_wkbs, "geometry", "value");
        traverse_within_and_mark(&index, probe_boxed, &[(0, 0), (0, 1), (1, 2)]);

        // The empty polygon was never assigned a geo_id, so it cannot appear as
        // unvisited. All 3 valid geo_ids are matched.
        let unvisited = index.unvisited_positions().unwrap();
        assert!(unvisited.is_empty());
    }
}
