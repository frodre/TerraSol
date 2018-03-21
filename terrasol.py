"""
Author: Andre Perkins

Plot creation for the planet and star system.
"""

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div, Slider
from bokeh.layouts import WidgetBox

import pandas
import numpy as np
import os

file_dir = os.path.dirname(__file__)
file_fullpath = os.path.join(file_dir, 'stellar_color_df.pckl')
star_color_data = pandas.read_pickle(file_fullpath)

AU_IN_M = 149597870700  # meters
LUMINOSITY_OUR_SUN = 3.828e26  # watts

# Styling for text table columns
left_style = '"text-align:left; font-weight:bold"'
right_style = '"text-align:right; font-weight:normal;"'
title_style = '"text-align:left; font-weight:bold; font-size:18px"'

class TerraSol(object):
    """Container for important functions and pieces of the stellar and planetary
    body web application."""

    def __init__(self, t_eff=6000, rel_luminosity=1, rel_planet_dist=1,
                 rel_planet_radius=1, plot_width=800, plot_height=400):

        # Calculate Star and planet characteristics
        stellar_radius = calc_star_radius(rel_luminosity, t_eff)
        planet_radius = calc_planet_radius_in_au(rel_planet_radius)
        xrange, yrange = get_plot_range(stellar_radius, rel_planet_dist)
        star_color = get_star_color(t_eff)
        star_E_out = calc_star_energy_flux(t_eff)
        planet_E_in = calc_planet_energy_in(rel_luminosity, rel_planet_dist)

        # Create star and planet figure
        p = figure(x_range=xrange, y_range=yrange, plot_width=plot_width,
                   plot_height=plot_height, toolbar_location='above',
                   tools='pan,wheel_zoom,reset')

        p.xaxis.axis_label = 'Distance (AU)'
        p.yaxis.axis_label = 'Distance (AU)'
        p.title.text = 'TerraSol Simulator'

        # Star Data Source
        star_data = ColumnDataSource(data=dict(name=['Sol'],
                                              radius=[stellar_radius],
                                              color=[star_color],
                                              xvalues=[0],
                                              yvalues=[0],
                                              line_color=[None],
                                              energy_out=[star_E_out],
                                              T_eff=[t_eff],
                                              luminosity=[rel_luminosity]))
        # planet data source
        planet_data = ColumnDataSource(data=dict(name=['Terra'],
                                                radius=[planet_radius],
                                                color=['Tan'],
                                                xvalues=[rel_planet_dist],
                                                yvalues=[0],
                                                line_color=['#ADD8E6'],  # light blue
                                                energy_in=[planet_E_in]))

        # Calculate invisible radius factor for hover tooltips
        invis_radius = radius_fix_factor(plot_width, plot_height,
                                         np.diff(xrange)[0],
                                         stellar_radius,
                                         np.diff(yrange)[0])
        sol, sol_invis = plot_body(p, star_data, invis_radius)
        terra, terra_invis = plot_body(p, planet_data, invis_radius)

        hover = HoverTool(renderers=[sol_invis],
                          tooltips=[('name', '@name'),
                                    ('T_eff (K)', '@T_eff{0,0}'),
                                    ('Energy Output (W/m^2)', '@energy_out{0,0.00}')])
        hover_terra = HoverTool(renderers=[terra_invis],
                                tooltips=[('Name', '@name'),
                                          ('Energy Input (W/m^2)', '@energy_in{0,0.00}')])
        p.add_tools(hover)
        p.add_tools(hover_terra)
        p.background_fill_color = '#190c26'  # midnight blue
        p.xgrid.visible = False
        p.ygrid.visible = False

        self.plot = p
        self.star_data = star_data
        self.planet_data = planet_data

        # Create informational DIV Elements for page
        star_text = self.create_star_text_html()
        planet_text = self.create_planet_text_html()

        star_div = Div(text=star_text, width=266, height=100)
        planet_div = Div(text=planet_text, width=266, height=100)
        empty_div = Div(width=266, height=100)

        self.star_div = star_div
        self.planet_div = planet_div
        self.div_row = [self.star_div, empty_div, self.planet_div]
        self.sliders = self.init_slider_wx()

    def get_planet_energy_in(self):
        return self.planet_data.data['energy_in'][0]

    def update_star(self, new_luminosity=None, new_t_eff=None):
        if new_luminosity is None:
            new_luminosity = self.star_data.data['luminosity'][0]

        if new_t_eff is None:
            new_t_eff = self.star_data.data['T_eff'][0]

        new_radius = calc_star_radius(new_luminosity, new_t_eff)
        new_color = get_star_color(new_t_eff)
        new_energy_out = calc_star_energy_flux(new_t_eff)

        star_update = dict(radius=[new_radius],
                           color=[new_color],
                           T_eff=[new_t_eff],
                           luminosity=[new_luminosity],
                           energy_out=[new_energy_out])

        self.star_data.data.update(star_update)
        new_star_text = self.create_star_text_html()
        self.star_div.text = new_star_text

        self.update_planet(new_star_luminosity=new_luminosity)

    def update_planet(self, radius=None, dist=None, new_star_luminosity=None):

        if radius is None:
            radius = self.planet_data.data['radius'][0]
        else:
            radius = calc_planet_radius_in_au(radius)

        if dist is None:
            dist = self.planet_data.data['xvalues'][0]

        if new_star_luminosity is None:
            star_luminosity = self.star_data.data['luminosity'][0]
        else:
            star_luminosity = new_star_luminosity

        energy_in = calc_planet_energy_in(star_luminosity, dist)

        planet_update = dict(radius=[radius], xvalues=[dist], energy_in=[energy_in])
        self.planet_data.data.update(planet_update)

        new_planet_text = self.create_planet_text_html()
        self.planet_div.text = new_planet_text

    def create_planet_text_html(self):

        name = self.planet_data.data['name'][0]
        dist = self.planet_data.data['xvalues'][0]
        radius = self.planet_data.data['radius'][0]
        energy_in = self.planet_data.data['energy_in'][0]
        star_radius = self.star_data.data['radius'][0]

        radius *= AU_IN_M
        dist = (dist - star_radius) * AU_IN_M

        text = """
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
                   </table>""".format(title_style=title_style,
                                      name=name,
                                      left_style=left_style,
                                      right_style=right_style,
                                      dist=dist,
                                      radius=radius,
                                      energy_in=energy_in)

        return text

    def create_star_text_html(self):

        name = self.star_data.data['name'][0]
        luminosity = self.star_data.data['luminosity'][0]
        t_eff = self.star_data.data['T_eff'][0]
        radius = self.star_data.data['radius'][0]
        energy_out = self.star_data.data['energy_out'][0]

        luminosity *= LUMINOSITY_OUR_SUN
        radius *= AU_IN_M

        text = """
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
                </table>""".format(title_style=title_style,
                                   name=name,
                                   left_style=left_style,
                                   right_style=right_style,
                                   t_eff=t_eff,
                                   luminosity=luminosity,
                                   radius=radius,
                                   energy_out=energy_out)

        return text

    def _t_eff_handler(self, attr, old, new):
        self.update_star(new_t_eff=new)

    def _luminosity_handler(self, attr, old, new):
        luminosity = 10**new
        self.update_star(new_luminosity=luminosity)

    def _planet_radius_handler(self, attr, old, new):
        self.update_planet(radius=new)

    def _planet_dist_handler(self, attr, old, new):
        dist = 10**new
        self.update_planet(dist=dist)

    def init_slider_wx(self):
        t_eff_slider = Slider(start=1000, end=40000, step=100, value=6000,
                              title='Star Effective Temperature (K)')
        luminosity_slider = Slider(start=-5, end=5, step=0.1, value=0,
                                   title='Relative Luminosity (10^x, solar units)')
        planet_radius_slider = Slider(start=0.1, end=4, step=0.1, value=1,
                                      title='Relative Planet Radius (earth radii)')
        planet_distance_slider = Slider(start=-1, end=3, step=0.1, value=0,
                                        title='Relative Planet Distance (10^x, AU)')

        t_eff_slider.on_change('value', self._t_eff_handler)
        luminosity_slider.on_change('value', self._luminosity_handler)
        planet_radius_slider.on_change('value', self._planet_radius_handler)
        planet_distance_slider.on_change('value', self._planet_dist_handler)

        star_wx = WidgetBox(t_eff_slider, luminosity_slider)
        planet_wx = WidgetBox(planet_radius_slider, planet_distance_slider)

        return [star_wx, planet_wx]


def plot_body(fig_handle, data_source, invis_radius):
    body = fig_handle.circle(x='xvalues', y='yvalues', radius='radius',
                             radius_dimension='y', fill_color='color',
                             line_color='line_color', source=data_source)

    invis = fig_handle.circle(x='xvalues', y='yvalues', radius=invis_radius,
                              radius_dimension='x', fill_color='color',
                              line_color='line_color', source=data_source,
                              alpha=0.0)

    return body, invis


# Plotting Help
def get_plot_range(star_radius, planet_dist):

    ydist = 3 * star_radius
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


# Star Functions
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
