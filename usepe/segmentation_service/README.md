# AirspaCells

Divide a regions airspace (U-space) into rectangular cells based on definable rules.

## Process

1. Load a [set of rules](config/rules.json) defining which ground classes and airspace types should be considered and how they impact the cells (i.e. by attaching limitations on who can enter, performance requirements, allowed altitude ranges and capacity factors to them)
2. Prepare a region that shall be segmented, this is done e.g. by supplying its name and automatically downloading the region's bounds (using OSM geocoding), then wrap these bounds into a rectilinear `Shapely` polygon
3. Gather the ground data for this region from the OSM API, identify features of interest and wrap them into categorized, rectilinear cells according to the ruleset
4. Repeat the above step for the airspace data for this region (supplied as a [OpenAIR format file](data/airspace/de_asp.txt))
5. Dissolve neighboring and intersecting cells of the same class into combined polygons
6. Overlapping cells are checked and modified, so that only the most severe cell class (going by the defined order in the ruleset) remains for each point
7. Everything that is not covered by now is designated as a single polygon with the lowest severity class, the existing cells are left out as holes in that polygon
8. All polygons are dissected into the minimum amount of rectangles

## Relevant U-space services

To perform the airspace segmentation, a number of planned U-space services would be needed. As these do not exist yet, simplified stubs are included in this tool that replace them by providing the required data from publicly available sources. The implemented services (and my understanding how they interact with each other and the airspace segmentation service) are:

-   Geo-awareness (**partially implemented**): takes the input data from the services below and process them to inform other services (here the airspace segmentation) and operators
-   Geospatial information (**partially implemented**): two-dimensional geo data is taken from OSM and used as a static input to identify ground features of interest
-   Population density (**to be added**): is not really considered right now, would greatly enhance the ground risk classes for the identified OSM features
-   Drone aeronautical information (**partially implemented**): a snapshot of the current airspace structure is taken from OpenAIP data and used as a static input
-   Dynamic geofencing (**to be added**): geofences should be added as a dynamic input to update the impacted cells
-   Weather information (**to be added**): wind turbulence hot spots should be added dynamically to geofence them or adapt the capacity factor and performance requirements

## Requirements

A set of requirements to be fulfilled by the airspace segmentation service to serve the `BlueSky` simulator's needs have been identified by _NOMMON_ and _DLR_:

-   [ ] Static segmentation of a given region
    -   [x] **Input** in a suitable format: region definition
        -   [x] Format: String used for geocoding, returns the first matching area from _OSM_
        -   [x] Format: `GeoDataFrame`, uses the first geometry in the frame (which is buffered and simplified)
    -   [x] Initial set using the least amount segments
    -   [x] Suitable 2D geometry for the layering / path planning approach: _Rectangular_
    -   [x] Consider pre-existing conditions and features
        -   [x] Air space structure: _OpenAIR_ zones (see the [rule definitions](config/rules.json))
        -   [x] Ground features: _OSM_ features analogous to _DFS_ [DIPUL](https://uas-betrieb.dfs.de/homepage/de/informationen/geografische-gebiete/) (see the [rule definitions](config/rules.json))
        -   [x] Wind field: interpolated forecast
    -   [x] Categorize resulting segments
    -   [x] **Output** in a suitable format: `DataFrame` as `GeoJSON`
        -   [x] Format: For the `DataFrame` specification see the [Examples section](#examples)
    -   [ ] _BlueSky_: attach properties / meaning to the arbitrary categories
-   [ ] Dynamic updates of a given set of segments
    -   [ ] **Input** in a suitable format: `DataFrame` as Python function argument
        -   [ ] Format: For the `DataFrame` specification see the [WP4 User Manual](#examples)
        -   [x] Wind data in .nc format importable using xarray package
        -   [ ] Last five minutes history of all planes location as numpy array or `DataFrame`
        -   [ ] Event specification in .geojson or `GeoDataFrame` including start and end times of the events and area of the event as polygon
        -   [ ] List of conflicts
        -   [ ] Flight plan of all drones and location of corresponding waypoints
    -   [ ] Consider changing conditions
        -   [x] Wind fields
        -   [x] Traffic density
        -   [x] Conflict rates
        -   [x] Emergency Flights
        -   [ ] Dynamic Geofences
    -   [x] Implement a rule engine that parses user defined rules (in a human readable format) and applies actions based on the values of provided `DataFrame` columns
        -   [ ] Format: JSON, see the [rule definitions](config/rules.json)
    -   [ ] Perform one of multiple possible actions on a given segment
        -   [x] Update geometric definitions by dividing or merging cells
            -   [ ] `insert_cell` ?
            -   [x] `join_cell`
            -   [x] `split_cell`
        -   [x] Update cells to assure required aspect ratio
        -   [x] Update cell behaviour: attach modifiers (for e.g. the max. or min. velocity or entry restrictions) to any cell
            -   [x] Format: Python Dictionary, `{ "modifier": None, "value": None, "factor": None, "timeout": None }`
    -   [x] **Output** in a suitable format: `DataFrame` as Python function return value
        -   [x] Format: enhanced version of the input frame with added `modifiers` column

## Usage

### Initial set of cells

Run the `segmentation.py` file to perform the initial (offline) segmentation. This automatically saves the cells into a `cells.geojson` (which is an exported `GeoDataFrame`) in your current working directory.

### Read cell data

To load this file back into a `GeoDataFrame` use:

```python
cells = gpd.read_file("cells.geojson", driver="GeoJSON"))
```

### Divide a cell

To divide a cell at the index `idx` in the `DataFrame` `cells` run:

```python
cells = split_cell(cells, idx, splitter)
```

where splitter can either be `"x"`, `"y"` (both dividing the cell in half for the given direction), or any `Shapely` geometry.

### Split cells to assure minimal required aspect ratio

```python
cells = split_aspect_ratio(cells, aspectRatio)
```

## Open Issues

-   <del>The dissection of the encompassing, lowest class, cell seems to create overlapping cells in a few places, so either the dissection algorithm needs to be more robust or a post-processing step could be added to clean up the rectangles</del>

## Examples

The cells below were created for the city of Braunschweig:

![Cells created for Braunschweig](docs/examples/braunschweig.png)

The cells bellow were created by updating initial set of cells for the city of Braunschweig assuring minimal aspect ratio 1:2

![Cells created for Braunschweig](docs/examples/braunschweig_updated.png)

Generally, the DataFrames contain two columns `parent` and `children` that basically make up a linked list of how a cell was divided. If `children` is not `None`, the `parent` cell is not active anymore and has been replaced by its divided children. So for example if a cell has been subdivided using `cells = update.split(cells, 42, 'x')` and we select the parent and all its children using `cells.loc[[42] + cells.loc[42, 'children']]` we receive the following view:

|      | class | floor | ceiling | parent | children     | geometry                        |
| ---: | :---- | ----: | ------: | -----: | :----------- | :------------------------------ |
|   42 | white |     0 |     500 |        | [1221, 1222] | POLYGON ((9.7... 52.4..., ...)) |
| 1221 | white |     0 |     500 |     42 |              | POLYGON ((9.7... 52.4..., ...)) |
| 1222 | white |     0 |     500 |     42 |              | POLYGON ((9.7... 52.4..., ...)) |

## TODO

-   [x] Create initial set of cells (offline)
-   [x] Consider micro-weather in the cell creation process
-   [ ] Provide methods to update cells with minimal impact if e.g. geofences are introduced
-   [x] Provide methods to dynamically split and join cells to adapt their size to traffic demand
-   [ ] Check environmental protection classes (see https://wiki.openstreetmap.org/wiki/DE:Key:protect_class and https://wiki.openstreetmap.org/wiki/DE:Key:protection_title)
