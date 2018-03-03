import numpy as np
from bokeh.plotting import figure, curdoc, show
from bokeh.layouts import layout, WidgetBox
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Div, Slider
from bokeh.models.tools import HoverTool
import pandas

star_color_data = pandas.read_pickle('stellar_color_df.pckl')
AU_IN_M = 149597870700  # meters
LUMINOSITY_OUR_SUN = 3.828e26  # watts
left_style = '"text-align:left; font-weight:bold"'
right_style = '"text-align:right; font-weight:normal;"'
title_style = '"text-align:left; font-weight:bold; font-size:18px"'


def get_plot_range(stellar_radius, planet_dist):

    ydist = 3 * stellar_radius
    yrange = (-ydist, ydist)
    x_buffer = planet_dist * 0.1
    xrange = (0 - x_buffer, planet_dist + x_buffer)

    return xrange, yrange


def radius_fix_factor(pwidth, pheight, total_xrange, y_radius, total_yrange):
    px_per_x = pwidth / total_xrange
    apparent_yradius_px = y_radius / total_yrange * pheight
    real_y_px = y_radius * px_per_x
    infl_factor = apparent_yradius_px / real_y_px

    equiv_radius_x = y_radius * infl_factor

    return equiv_radius_x


def calc_star_radius(rel_luminosity, t_eff):

    L = rel_luminosity * LUMINOSITY_OUR_SUN
    bb_output = calc_star_energy_flux(t_eff)
    radius = np.sqrt(L / (4 * np.pi * bb_output))

    # Return normalized radius in AU
    return radius / AU_IN_M


def calc_star_energy_flux(t_eff):
    sigma = 5.670367e-8  # Stefan-Boltzmann Constant W.m^-2.K^-4
    return sigma * t_eff ** 4


def calc_planet_radius_in_au(relative_radius):
    earth_radius = 6.371e6  # meters
    radius = relative_radius * earth_radius
    return radius / AU_IN_M


def calc_planet_energy_in(rel_luminosity, rel_planet_dist):
    luminosity_our_sun = 3.828e26  # watts
    luminosity = rel_luminosity * luminosity_our_sun
    toa_energy = luminosity / (4 * np.pi * (rel_planet_dist * AU_IN_M) ** 2)
    return toa_energy


def get_star_color(t_eff):
    return star_color_data['hexrgb'].loc[t_eff, '10deg']


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

    stellar_radius = calc_star_radius(rel_luminosity, t_eff)
    planet_radius = calc_planet_radius_in_au(rel_planet_radius)
    xrange, yrange = get_plot_range(stellar_radius, rel_planet_dist)
    star_color = get_star_color(t_eff)
    star_E_out = calc_star_energy_flux(t_eff)
    planet_E_in = calc_planet_energy_in(rel_luminosity, rel_planet_dist)

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

    invis_radius = radius_fix_factor(plot_width, plot_height,
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


def create_star_text_html(sol_data):

    name = sol_data.data['name'][0]
    luminosity = sol_data.data['luminosity'][0]
    t_eff = sol_data.data['T_eff'][0]
    radius = sol_data.data['radius'][0]
    energy_out = sol_data.data['energy_out'][0]

    luminosity *= LUMINOSITY_OUR_SUN
    radius *= AU_IN_M

    text = f"""
            <table style="width:100%">
                <tr><th colspan="2" style={title_style}>{name} Characteristics</th></tr>
                <tr>
                    <td style={left_style}>Luminosity: </td>
                    <td style={right_style}>{luminosity:1.4e} W</td>
                </tr>
                <tr>
                    <td style={left_style}>Effective Temperature:</td>
                    <td style={right_style}> {t_eff:} K</td>
                </tr>
                <tr>
                    <td style={left_style}>Radius: </td>
                    <td style={right_style}>{radius:1.4e} m</td>
                </tr>
                <tr>
                    <td style={left_style}>Energy Flux:</td>
                    <td style={right_style}>{energy_out:1.4e} W/m^</td>
                </tr>
            </table>"""

    return text


def create_planet_text_html(terra_data, stellar_radius):

    name = terra_data.data['name'][0]
    dist = terra_data.data['xvalues'][0]
    radius = terra_data.data['radius'][0]
    energy_in = terra_data.data['energy_in'][0]

    radius *= AU_IN_M
    dist = (dist - stellar_radius) * AU_IN_M

    text = f"""
            <table style="width:100%">
                <tr><th colspan="2" style={title_style}>{name} Characteristics </th></tr>
                <tr>
                    <td style={left_style}>Distance from star:</td>
                    <td style={right_style}>{dist:1.4e} m</td>
                </tr>
                <tr>
                    <td style={left_style}>Radius:</td>
                    <td style={right_style}>{radius:1.4e} m</td>
                </tr>
                <tr>
                    <td style={left_style}>Energy Flux In: </td>
                    <td style={right_style}>{energy_in:1.4e} W/m^2</td>
                </tr>
            </table>"""

    return text


# Initialize Star and Planet
p, star_data, planet_data = init_star_planet_plot()

star_text = create_star_text_html(star_data)
planet_text = create_planet_text_html(planet_data, star_data.data['radius'][0])

# Create Elements for page
star_div = Div(text=star_text, width=266, height=100)
planet_div = Div(text=planet_text, width=266, height=100)
empty_div = Div(width=266, height=100)


def update_star_info(new_luminosity=None, new_t_eff=None):
    if new_luminosity is None:
        new_luminosity = star_data.data['luminosity'][0]

    if new_t_eff is None:
        new_t_eff = star_data.data['T_eff'][0]

    new_radius = calc_star_radius(new_luminosity, new_t_eff)
    new_color = get_star_color(new_t_eff)
    new_energy_out = calc_star_energy_flux(new_t_eff)

    star_update = dict(radius=[new_radius],
                       color=[new_color],
                       energy_out=[new_energy_out])

    star_data.data.update(star_update)
    new_star_text = create_star_text_html(star_data)
    star_div.text = new_star_text

    update_planet_info(new_radius, new_star_luminosity=new_luminosity)

def update_planet_info(star_radius, radius=None, dist=None, new_star_luminosity=None):

    if radius is None:
        radius = planet_data.data['radius'][0]
    else:
        radius = calc_planet_radius_in_au(radius)

    if dist is None:
        dist = planet_data.data['xvalues'][0]

    if new_star_luminosity is None:
        star_luminosity = star_data.data['luminosity'][0]
    else:
        star_luminosity = new_star_luminosity

    energy_in = calc_planet_energy_in(star_luminosity, dist)

    planet_update = dict(radius=[radius], xvalues=[dist], energy_in=[energy_in])
    planet_data.data.update(planet_update)

    new_planet_text = create_planet_text_html(planet_data, star_radius)
    planet_div.text = new_planet_text

    new_xrange, new_yrange = get_plot_range(star_radius, dist)
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

widgets = WidgetBox(t_eff_slider, luminosity_slider)
planet_widgets = WidgetBox(planet_radius_slider, planet_distance_slider)
plot_layout = layout([[p],
                      [star_div, empty_div, planet_div],
                      [widgets, planet_widgets]],
                     sizing_mode='fixed')
# show(plot_layout)
curdoc().add_root(plot_layout)
curdoc().title = 'TerraSol'