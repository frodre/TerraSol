import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh

# import holoviews as hv
# import holoviews.plotting.bokeh
# import geoviews as gv
# import geoviews.feature as gf
# import cartopy.crs as ccrs


# hv.extension('bokeh')
# renderer = hv.renderer('bokeh')
# hv.opts("Overlay [width=600 height=500] Image (cmap='viridis') Feature (line_color='black')")
# hv.output(size=200)

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

renderer = hv.renderer('bokeh')

points = hv.Points(np.random.randn(1000,2 )).opts(plot=dict(tools=['box_select', 'lasso_select']))
selection = hv.streams.Selection1D(source=points)

def selected_info(index):
    arr = points.array()[index]
    if index:
        label = 'Mean x, y: %.3f, %.3f' % tuple(arr.mean(axis=0))
    else:
        label = 'No selection'
    return points.clone(arr, label=label).opts(style=dict(color='red'))

layout = points + hv.DynamicMap(selected_info, streams=[selection])

doc = renderer.server_doc(layout)
doc.title = 'HoloViews App'