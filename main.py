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
# p, star_data, planet_data = terrasol.init_star_planet_plot()

# Initialize Climate Model
planet_energy_in = terrasol_ux.planet_data.data['energy_in'][0] / 4
planet_climate = climate.EnergyBalanceModel(Q=planet_energy_in)
climate_result = planet_climate.solve_climate()
final_T_dataframe = planet_climate.convert_1d_to_grid()

# Climate inputs
planet_emiss = TextInput(title='Planetary IR energy out (W/m^2)',
                         value='{:.2f}'.format(planet_climate.A))
planet_atm_forcing = TextInput(title='Atmosphere IR adjustment (W/m^2)',
                               value='{:.1f}'.format(planet_climate.B))
solar_input = TextInput(title='Incoming solar (W/m^2) [Divided by 4]',
                        value='{:.2f}'.format(planet_energy_in))
energy_transport = TextInput(title='Energy transport towards poles (1/C)',
                             value='{:.1f}'.format(planet_climate.D))
s2_input = TextInput(title='S2 (what is this for?)',
                     value='{:.3f}'.format(planet_climate.S2))
heat_capacity = TextInput(title='Planetary heat capacity (C/yr)',
                          value='{:.1f}'.format(planet_climate.C))
numlats = Slider(start=40, end=180, step=1, value=70,
                 title='Number of latitudes in model')
init_planet_T = Select(title='Initial planet temperature',
                       value='normal',
                       options=['normal', 'warm', 'cold'])
calc_climate = Button(label='Simulate Climate', button_type='success')


def sim_climate_handler():
    A = float(planet_emiss.value.strip())
    B = float(planet_atm_forcing.value.strip())
    Q = float(solar_input.value.strip())
    D = float(energy_transport.value.strip())
    S2 = float(s2_input.value.strip())
    C = float(heat_capacity.value.strip())
    nlats = int(numlats.value)
    init_condition = init_planet_T.value

    planet_climate = climate.EnergyBalanceModel(A=A, B=B, Q=Q, D=D,
                                                S2=S2, C=C, nlats=nlats,
                                                init_condition=init_condition)
    res = planet_climate.solve_climate()
    print(res)

calc_climate.on_click(sim_climate_handler)

climate_inputs1 = WidgetBox(children=[planet_emiss,
                                      planet_atm_forcing,
                                      solar_input],
                            width=int(800/3))
climate_inputs2 = WidgetBox(energy_transport, s2_input, heat_capacity)
model_inputs = WidgetBox(numlats, init_planet_T)

plot_layout = layout([[terrasol_ux.plot],
                      terrasol_ux.div_row,
                      terrasol_ux.sliders,
                      [climate_inputs1, climate_inputs2, model_inputs],
                      [calc_climate]],
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
