import numpy as np
from bokeh.plotting import figure, curdoc, show
from bokeh.layouts import layout, WidgetBox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Div, Slider, TextInput, Select
from bokeh.models.tools import HoverTool

import utils
import climate

AU_IN_M = utils.AU_IN_M
LUMINOSITY_OUR_SUN = utils.LUMINOSITY_OUR_SUN


def plot_body(fig_handle, data_source, invis_radius):
    body = fig_handle.circle(x='xvalues', y='yvalues', radius='radius',
                             radius_dimension='y', fill_color='color',
                             line_color='line_color', source=data_source)

    invis = fig_handle.circle(x='xvalues', y='yvalues', radius=invis_radius,
                              radius_dimension='x', fill_color='color',
                              line_color='line_color', source=data_source,
                              alpha=0.0)

    return body, invis


def init_star_planet_plot(t_eff=6000, rel_luminosity=1,
                          rel_planet_dist=1, rel_planet_radius=1,
                          plot_width=800,
                          plot_height=400):

    stellar_radius = utils.calc_star_radius(rel_luminosity, t_eff)
    planet_radius = utils.calc_planet_radius_in_au(rel_planet_radius)
    xrange, yrange = utils.get_plot_range(stellar_radius, rel_planet_dist)
    star_color = utils.get_star_color(t_eff)
    star_E_out = utils.calc_star_energy_flux(t_eff)
    planet_E_in = utils.calc_planet_energy_in(rel_luminosity, rel_planet_dist)

    p = figure(x_range=xrange, y_range=yrange, plot_width=plot_width,
               plot_height=plot_height, toolbar_location='above',
               tools='pan,wheel_zoom,reset')

    # Star Data Source
    sol_data = ColumnDataSource(data=dict(name=['Sol'],
                                          radius=[stellar_radius],
                                          color=[star_color],
                                          xvalues=[0],
                                          yvalues=[0],
                                          line_color=[None],
                                          energy_out=[star_E_out],
                                          T_eff=[t_eff],
                                          luminosity=[rel_luminosity]))

    terra_data = ColumnDataSource(data=dict(name=['Terra'],
                                            radius=[planet_radius],
                                            color=['Tan'],
                                            xvalues=[rel_planet_dist],
                                            yvalues=[0],
                                            line_color=['#ADD8E6'],  # light blue
                                            energy_in=[planet_E_in]))

    invis_radius = utils.radius_fix_factor(plot_width, plot_height,
                                           np.diff(xrange)[0],
                                           stellar_radius,
                                           np.diff(yrange)[0])
    sol, sol_invis = plot_body(p, sol_data, invis_radius)
    terra, terra_invis = plot_body(p, terra_data, invis_radius)

    hover = HoverTool(renderers=[sol_invis],
                      tooltips=[('name', '@name'),
                                ('T_eff (K)', '@T_eff{0,0}'),
                                ('Energy Output (W/m^2)', '@energy_out{0,0.00}')])
    hover_terra = HoverTool(renderers=[terra_invis],
                            tooltips=[('Name', '@name'),
                                      ('Energy Input (W/m^2)', '@energy_in{0,0.00}')])
    p.add_tools(hover)
    p.add_tools(hover_terra)
    p.background_fill_color = '#190c26'  #midnight blue
    p.xgrid.visible = False
    p.ygrid.visible = False

    return p, sol_data, terra_data


def init_planet_climate(energy_in):

    climate_model_inputs = ColumnDataSource(data=dict(A=211.22,
                                                      B=2.1,
                                                      Q=energy_in/4,
                                                      D=1.2,
                                                      S2=-0.482,
                                                      C=9.8,
                                                      nlats=70,
                                                      tol=1e-5,
                                                      init_condition='normal'))


# Initialize Star and Planet
p, star_data, planet_data = init_star_planet_plot()

star_text = utils.create_star_text_html(star_data)
planet_text = utils.create_planet_text_html(planet_data,
                                            star_data.data['radius'][0])

# Initialize Climate Model
planet_energy_in = planet_data.data['energy_in'][0] / 4
planet_climate = climate.EnergyBalanceModel(Q=planet_energy_in)
climate_result = planet_climate.solve_climate()

# Create Elements for page
star_div = Div(text=star_text, width=266, height=100)
planet_div = Div(text=planet_text, width=266, height=100)
empty_div = Div(width=266, height=100)


def update_star_info(new_luminosity=None, new_t_eff=None):
    if new_luminosity is None:
        new_luminosity = star_data.data['luminosity'][0]

    if new_t_eff is None:
        new_t_eff = star_data.data['T_eff'][0]

    new_radius = utils.calc_star_radius(new_luminosity, new_t_eff)
    new_color = utils.get_star_color(new_t_eff)
    new_energy_out = utils.calc_star_energy_flux(new_t_eff)

    star_update = dict(radius=[new_radius],
                       color=[new_color],
                       T_eff=[new_t_eff],
                       energy_out=[new_energy_out])

    star_data.data.update(star_update)
    new_star_text = utils.create_star_text_html(star_data)
    star_div.text = new_star_text

    update_planet_info(new_radius, new_star_luminosity=new_luminosity)


def update_planet_info(star_radius, radius=None, dist=None, new_star_luminosity=None):

    if radius is None:
        radius = planet_data.data['radius'][0]
    else:
        radius = utils.calc_planet_radius_in_au(radius)

    if dist is None:
        dist = planet_data.data['xvalues'][0]

    if new_star_luminosity is None:
        star_luminosity = star_data.data['luminosity'][0]
    else:
        star_luminosity = new_star_luminosity

    energy_in = utils.calc_planet_energy_in(star_luminosity, dist)

    planet_update = dict(radius=[radius], xvalues=[dist], energy_in=[energy_in])
    planet_data.data.update(planet_update)

    new_planet_text = utils.create_planet_text_html(planet_data, star_radius)
    planet_div.text = new_planet_text

    new_xrange, new_yrange = utils.get_plot_range(star_radius, dist)
    p.x_range.start, p.x_range.end = new_xrange
    # p.y_range.start, p.y_range.end = new_yrange


def t_eff_handler(attr, old, new):
    update_star_info(new_t_eff=new)


def luminosity_handler(attr, old, new):
    luminosity = 10**new
    update_star_info(new_luminosity=luminosity)


def planet_radius_handler(attr, old, new):
    update_planet_info(star_data.data['radius'][0],
                       radius=new)


def planet_dist_handler(attr, old, new):
    dist = 10**new
    update_planet_info(star_data.data['radius'][0],
                       dist=dist)


t_eff_slider = Slider(start=1000, end=40000, step=100, value=6000,
                      title='Star Effective Temperature (K)')
luminosity_slider = Slider(start=-5, end=5, step=0.1, value=0,
                           title='Relative Luminosity (10^x, solar units)')
planet_radius_slider = Slider(start=0.1, end=4, step=0.1, value=1,
                              title='Relative Planet Radius (earth radii)')
planet_distance_slider = Slider(start=-1, end=3, step=0.01, value=0,
                                title='Relative Planet Distance (10^x, AU)')

t_eff_slider.on_change('value', t_eff_handler)
luminosity_slider.on_change('value', luminosity_handler)
planet_radius_slider.on_change('value', planet_radius_handler)
planet_distance_slider.on_change('value', planet_dist_handler)

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


climate_inputs1 = WidgetBox(children=[planet_emiss,
                                      planet_atm_forcing,
                                      solar_input],
                            width=int(800/3))
climate_inputs2 = WidgetBox(energy_transport, s2_input, heat_capacity)
model_inputs = WidgetBox(numlats, init_planet_T)
widgets = WidgetBox(t_eff_slider, luminosity_slider)
planet_widgets = WidgetBox(planet_radius_slider, planet_distance_slider)
plot_layout = layout([[p],
                      [star_div, empty_div, planet_div],
                      [widgets, planet_widgets],
                      [climate_inputs1, climate_inputs2, model_inputs]],
                     sizing_mode='fixed')
show(plot_layout)
# curdoc().add_root(plot_layout)
# curdoc().title = 'TerraSol'