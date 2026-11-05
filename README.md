# spatialchunk

[![tests](https://github.com/maximedion2/spatialchunk/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/maximedion2/spatialchunk/actions/workflows/tests.yml)
[![lint](https://github.com/maximedion2/spatialchunk/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/maximedion2/spatialchunk/actions/workflows/lint.yml)

## Overview
This library is being designed to allow efficiently running spatial queries against chunked file storage, namely zarr and icechunk. In particular, it aims at handling spatial join pushdown against data that can be pruned via spatial indexing. It started off as a project mostly for reading zarr data, but has evolved into something that adds more focus on the geospatial queries.

Most of what I'm doing here is from scratch, simply so that I can experiment and see what works best, try different things, etc... which is also why it's a standalone libray and not a contribution to an existing project (see below for list of other relevant project).

Everything here is based on the [datafusion](https://github.com/apache/datafusion), [arrow-rs](https://github.com/apache/arrow-rs) and [zarrs](https://github.com/zarrs/zarrs) crates.

## Supported data stores and features

This library supports zarr v3 and icechunk format. I will write out a more detailed list of supported data schemas and store structure impact on query performance.

## Current state

My plan is to build a rust crate and python package, with all the logic being in rust and the python package just being a light wrapper around the rust code. Right now though, I have yet to publish anything, and I first need to decide on what gets exposed, after which I will write up extensive documentation before publishing. I do have a lot of the function logic right now, arguably the core goal of this library has been achieved, I do need to work on more extensive testing though.

## Example

See full example [here](crates/spatialchunk/examples/overture_era5_join.rs)

## Roadmap

### Reading zarr/icechunk data
- [x] Reading from zarr v3
- [x] Reading from icechunk
- [x] 1D coordinate handling
- [ ] Discovery of groups, arrays (mapping to tables, columns, etc...)
- [x] Projection pushdown
- [x] Filter pushdown (coordinate or not)
- [x] Store level statistics
- [x] Read/decompress metrics
- [x] Limits
- [x] Attribute based transformations (in particular, handling timestamp types)
- [x] Parallelism via partitioning
- [x] I/O and compute runtime split
- [x] AWS s3 support
- [x] Goggle GCS support
- [ ] extend support beyond zarr and icechunk (virtual chunks?)

### Geospatial functionalities
- [x] On-the-fly spatial indexing
- [x] Spatial join logic and udfs for common predicates
- [ ] Spatial join logic and udfs for less common predicates
- [x] ST_Area, ST_Distance
- [x] ST_Point
- [ ] ST_Buffer
- [ ] ST_Intersections
- [ ] K_Nearest
- [x] WKB and WKT support
- [x] Optimizer logic to to convert NJLs to properly optimized spatial joins
- [ ] Geoparquet 1.1 native GeoArrow encoding support
- [x] Spatial join pushdown to geoparquet files
- [ ] Disk dumps when running out of ram
- [ ] Support for partially materializing the build side 

### Chunks + geospatial
- [x] Spatial join pushdown against zarr/icechunk (only ST_Point for zarr/icecehunk)
- [ ] Support for spatial join pushdown on more complex geo transformations

## Other relevant projects

The main geospatial query rust (and python) engine is [sedona-db](https://github.com/apache/sedona-db). As mentioned above, I'm writing my own (much more limited) engine to test out different things, possibly implement zarr/icechunk specialized features, withotu cluttering existing projects all the things I want to try. I have no intentions of making the geospatial part of this library as polished and complete as sedona-db, it's already a perfectly good library.

The are several other efforts to extend spatial queries to zarr and icechunk, here's a couple that I know about:
- [zarr-datafusion](https://github.com/stratoscale-io/zarr-datafusion)
- [xarray-sql](https://github.com/xqlsystems/xarray-sql)

I think the icechunk library/package itself now supports some SQL functionalities as well, I believe it's ongoing work.

## Contributing

I don't have any official guidelines for now, if you're interested in making contributions, or you have use cases that would benefit from this library and want to test it, just reach out!