from bokeh.plotting import curdoc, show
from bokeh.layouts import layout

import terrasol
import climate


# Initialize Star and Planet
terrasol_ux = terrasol.TerraSol()

# # Initialize Climate Model
# energy_in = terrasol_ux
# climate_ux = climate.PlanetClimateEBM(terrasol_ux)
# print(climate_ux.climate_result)

# Initialize Simpler Climate Model
simple_climate_ux = climate.SimpleClimate(terrasol_ux)

plot_layout = layout([[terrasol_ux.title_div],
                      [terrasol_ux.plot],
                      terrasol_ux.div_row,
                      terrasol_ux.sliders,
                      [simple_climate_ux.title_div],
                      [simple_climate_ux.plot],
                      [simple_climate_ux.terra_div],
                      simple_climate_ux.model_wx],
                     sizing_mode='fixed')

# show(plot_layout)
curdoc().add_root(plot_layout)
curdoc().title = 'TerraSol'


