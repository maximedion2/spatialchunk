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

pub(crate) mod file_source;
pub(crate) mod geos;
pub(crate) mod join_predicates;
pub(crate) mod probe_pruning_expr;
pub(crate) mod spatial_join_exec;
pub(crate) mod spatial_join_optimizer;
pub(crate) mod spatial_join_stream;
pub mod udfs;

pub use spatial_join_exec::SpatialJoinExec;
pub use spatial_join_optimizer::SpatialJoinPhysicalOptimizer;
pub use udfs::{
    StAreaUdf, StBufferUdf, StContainsUdf, StCoveredByUdf, StCoversUdf, StDWithinUdf,
    StDistanceUdf, StGeomFromWkbUdf, StGeomFromWktUdf, StIntersectsUdf, StPointUdf, StTouchesUdf,
    StWithinUdf,
};

// #[cfg(test)]
pub(crate) mod test_utils;
