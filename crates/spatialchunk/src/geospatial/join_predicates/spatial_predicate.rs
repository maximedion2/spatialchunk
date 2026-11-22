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

use datafusion::physical_plan::PhysicalExpr;

#[derive(Debug, Clone, PartialEq)]
pub enum SpatialRelationType {
    Within,
    Contains,
    CoveredBy,
    Covers,
    Intersects,
    Touches,
    DWithin { distance: f64 },
}

#[derive(Debug, Clone)]
pub struct RelationPredicate {
    pub left: Arc<dyn PhysicalExpr>,
    pub right: Arc<dyn PhysicalExpr>,
    pub relation_type: SpatialRelationType,
    pub is_geography: bool,
}

impl SpatialRelationType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "st_within" => Some(Self::Within),
            "st_contains" => Some(Self::Contains),
            "st_coveredby" => Some(Self::CoveredBy),
            "st_covers" => Some(Self::Covers),
            "st_intersects" => Some(Self::Intersects),
            "st_touches" => Some(Self::Touches),
            _ => None,
        }
    }

    // The relation with its operands swapped, e.g. a within b is the same as
    // b contains a. Used to express a predicate whose arguments reference the
    // join sides in reverse by swapping the arguments.
    pub fn opposite(&self) -> Self {
        match self {
            Self::Within => Self::Contains,
            Self::Contains => Self::Within,
            Self::CoveredBy => Self::Covers,
            Self::Covers => Self::CoveredBy,
            // Intersects, Touches and DWithin are symmetric, so swapping operands
            // leaves them unchanged.
            Self::Intersects => Self::Intersects,
            Self::Touches => Self::Touches,
            Self::DWithin { distance } => Self::DWithin {
                distance: *distance,
            },
        }
    }
}

impl RelationPredicate {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        relation_type: SpatialRelationType,
    ) -> Self {
        Self {
            left,
            right,
            relation_type,
            is_geography: false,
        }
    }

    pub fn with_is_geography(mut self, is_geography: bool) -> Self {
        self.is_geography = is_geography;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name_and_opposite() {
        assert_eq!(
            SpatialRelationType::from_name("ST_Within"),
            Some(SpatialRelationType::Within)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_Contains"),
            Some(SpatialRelationType::Contains)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_CoveredBy"),
            Some(SpatialRelationType::CoveredBy)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_Covers"),
            Some(SpatialRelationType::Covers)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_Intersects"),
            Some(SpatialRelationType::Intersects)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_Touches"),
            Some(SpatialRelationType::Touches)
        );

        assert_eq!(
            SpatialRelationType::Within.opposite(),
            SpatialRelationType::Contains
        );
        assert_eq!(
            SpatialRelationType::Contains.opposite(),
            SpatialRelationType::Within
        );
        assert_eq!(
            SpatialRelationType::CoveredBy.opposite(),
            SpatialRelationType::Covers
        );
        assert_eq!(
            SpatialRelationType::Covers.opposite(),
            SpatialRelationType::CoveredBy
        );
        assert_eq!(
            SpatialRelationType::Intersects.opposite(),
            SpatialRelationType::Intersects
        );
        assert_eq!(
            SpatialRelationType::Touches.opposite(),
            SpatialRelationType::Touches
        );

        // DWithin is not produced by from_name (the distance comes from the call's
        // third argument), but it is symmetric and must preserve the distance.
        assert_eq!(SpatialRelationType::from_name("st_dwithin"), None);
        let dwithin = SpatialRelationType::DWithin { distance: 5.0 };
        assert_eq!(dwithin.opposite(), dwithin);
        match dwithin.opposite() {
            SpatialRelationType::DWithin { distance } => assert_eq!(distance, 5.0),
            other => panic!("expected DWithin, got {other:?}"),
        }
    }
}
