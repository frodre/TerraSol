from bokeh.plotting import curdoc
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

plot_layout = layout([[terrasol_ux.plot],
                      terrasol_ux.div_row,
                      terrasol_ux.sliders,
                      [simple_climate_ux.plot],
                      # climate_ux.input_wx,
                      # [climate_ux.calc_button]
                      ],
                     sizing_mode='fixed')
# show(plot_layout)
curdoc().add_root(plot_layout)
curdoc().title = 'TerraSol'


