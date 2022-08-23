from datetime import datetime

import hvplot
import hvplot.pandas

from usepe.segmentation_service.python.lib2 import misc
import geopandas as gpd
import xarray as xr


if __name__ == "__main__":
    region = "Hannover"
    windFile = "test_hannover_1m_3d"

    cells = gpd.read_file( ( "usepe/segmentation_service/data/examples/" + region + ".geojson" ), driver="GeoJSON" )
    windData = xr.open_dataset( ( "usepe/segmentation_service/data/wind/test_hannover_1m/" + windFile + ".nc" ) )

    start = datetime.now()
    misc.update_wind( cells, windData, True )
    print( "updating wind data took", datetime.now() - start, "seconds" )

    plot = cells.hvplot( 
        c="class",
        geo=True,
        frame_height=1000,
        tiles="CartoDark",
        hover_cols=["z_min", "z_max", "capacity"],
        alpha=0.2,
    )
    misc.cells2file( cells, ( "usepe/segmentation_service/data/examples/" + region + "_wind_interpolated.geojson" ) )
    hvplot.show( plot )
