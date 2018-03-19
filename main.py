import numpy as np
from bokeh.plotting import figure, curdoc, show
from bokeh.layouts import layout, WidgetBox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Div, Slider, TextInput, Select, Button
from bokeh.models.tools import HoverTool

# import holoviews as hv
# import holoviews.plotting.bokeh
# import geoviews as gv
# import geoviews.feature as gf
# import cartopy.crs as ccrs

import terrasol
import climate

# hv.extension('bokeh')
# renderer = hv.renderer('bokeh')
# hv.opts("Overlay [width=600 height=500] Image (cmap='viridis') Feature (line_color='black')")
# hv.output(size=200)

# Initialize Star and Planet
terrasol_ux = terrasol.TerraSol()

# Initialize Climate Model
energy_in = terrasol_ux
climate_ux = climate.PlanetClimateEBM(terrasol_ux)
print(climate_ux.climate_result)

plot_layout = layout([[terrasol_ux.plot],
                      terrasol_ux.div_row,
                      terrasol_ux.sliders,
                      climate_ux.input_wx,
                      [climate_ux.calc_button]],
                     sizing_mode='fixed')
# show(plot_layout)
curdoc().add_root(plot_layout)
curdoc().title = 'TerraSol'

# xr_dataset = gv.Dataset(data=final_T_dataframe.to_dataset(name='T'),
#                         vdims=['T'],
#                         kdims=['lat', 'lon'],
#                         crs=ccrs.PlateCarree())
# temp_map = xr_dataset.to.image()
# playout = temp_map * gf.coastline()
#
# points = hv.Points(np.random.randn(1000,2 )).opts(plot=dict(tools=['box_select', 'lasso_select']))
# points2 = hv.Points(np.random.randn(1000,2 )).opts(plot=dict(tools=['box_select', 'lasso_select']))
# playout = points + points2
# doc = renderer.server_doc(playout)
# doc.title = 'Testing'
#
# # plot = renderer.get_plot(temp_map, curdoc())
# # show(layout([plot.state]))
