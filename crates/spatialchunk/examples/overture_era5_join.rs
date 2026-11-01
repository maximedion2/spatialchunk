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
use std::time::Instant;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::Result;
use spatialchunk::zarr_session_context;
use zarrs_object_store::object_store::aws::AmazonS3Builder;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = zarr_session_context()?;

    // county polygons from overture maps (parquet on s3). datafusion needs an
    // object store registered for the bucket to read it.
    let overture_bucket = "s3://overturemaps-us-west-2";
    let overture_store = AmazonS3Builder::new()
        .with_url(overture_bucket)
        .with_region("us-west-2")
        .with_skip_signature(true)
        .build()?;
    ctx.register_object_store(
        ListingTableUrl::parse(overture_bucket)?
            .object_store()
            .as_ref(),
        Arc::new(overture_store),
    );
    ctx.sql(
        "CREATE EXTERNAL TABLE division_areas
         STORED AS PARQUET
         LOCATION 's3://overturemaps-us-west-2/release/2026-09-23.0/theme=divisions/type=division_area/'",
    )
    .await?;

    // era5 reanalysis data from earthmover (an icechunk repo on s3).
    ctx.sql(
        "CREATE EXTERNAL TABLE era5
         STORED AS ICECHUNK_REPO
         LOCATION 's3://earthmover-icechunk-era5/icechunkV2'
         OPTIONS (
             'icechunk.group' 'single/temporal',
             's3.anonymous' 'true',
             's3.region' 'us-east-1'
         )",
    )
    .await?;

    // the average 2m temperature in each county around the bay area, on
    // 2024-01-01 at 10:00. longitudes go from 0 to 360, they're shifted to
    // -180 to 180 to match the overture geometries.
    let query = "
        WITH counties AS (
            SELECT names['primary'] AS name, st_geomfromwkb(geometry) AS geom
            FROM division_areas
            WHERE subtype = 'county'
                AND class = 'land'
                AND bbox['xmin'] < -118
                AND bbox['xmax'] > -121
                AND bbox['ymin'] < 39
                AND bbox['ymax'] > 36
        ),
        temperature AS (
            SELECT
                latitude AS lat,
                CASE WHEN longitude > 180 THEN longitude - 360 ELSE longitude END AS lon,
                t2m AS temperature
            FROM era5
            WHERE valid_time = TIMESTAMP '2024-01-01T10:00:00'
        )
        SELECT c.name, avg(t.temperature) AS avg_temperature
        FROM counties c
        JOIN temperature t
            ON st_within(st_point(t.lon, t.lat), c.geom)
        GROUP BY c.name
        ORDER BY c.name
    ";

    // evaluate the parquet filters while decoding, so the large columns (e.g. the
    // geometries) are only decoded for the few rows that pass.
    ctx.sql("SET datafusion.execution.parquet.pushdown_filters = true")
        .await?
        .collect()
        .await?;

    // run the query with and without the spatial join's dynamic filter, which
    // prunes the era5 chunks that can't match any county.
    //
    // I think most of the time is spent on reading data so it depends on connection
    // speed rather than cpu performance, but on an okay but not great connection
    // enabling the dynamic filter pushdown reduces the query time from ~2700 secs
    // to about ~35 secs.
    let mut results = Vec::new();
    for join_dynamic_filters in [true, false] {
        ctx.sql(&format!(
            "SET datafusion.optimizer.enable_join_dynamic_filter_pushdown = {join_dynamic_filters}"
        ))
        .await?
        .collect()
        .await?;

        let start = Instant::now();
        let batches = ctx.sql(query).await?.collect().await?;
        let elapsed = start.elapsed();

        let formatted = pretty_format_batches(&batches)?.to_string();
        println!("join dynamic filters: {join_dynamic_filters}, elapsed: {elapsed:?}");
        println!("{formatted}");
        results.push(formatted);
    }

    println!("results match: {}", results[0] == results[1]);

    Ok(())
}
