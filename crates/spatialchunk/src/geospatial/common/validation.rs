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

use arrow_schema::Field;
use datafusion::common::{plan_err, Result};

use super::constants::{ARROW_EXTENSION_NAME_KEY, GEOARROW_WKB_EXTENSION_NAME};

// Errors unless `field` is tagged as WKB geometry (the `geoarrow.wkb` extension
// name). Used to reject bare binary columns from geometry UDFs and the spatial
// join, forcing an explicit `st_geomfromwkb` / `st_geomfromwkt` tag first.
pub(crate) fn require_geometry(field: &Field) -> Result<()> {
    if is_geometry_field(field) {
        return Ok(());
    }
    plan_err!(
        "expected a geometry column tagged '{GEOARROW_WKB_EXTENSION_NAME}' \
         (use st_geomfromwkb / st_geomfromwkt), got a plain {} column",
        field.data_type()
    )
}

pub(crate) fn is_geometry_field(field: &Field) -> bool {
    field
        .metadata()
        .get(ARROW_EXTENSION_NAME_KEY)
        .is_some_and(|name| name == GEOARROW_WKB_EXTENSION_NAME)
}
