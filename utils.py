"""
Author: Andre Perkins
"""

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


# Update functions
def create_star_text_html(sol_data):

    name = sol_data.data['name'][0]
    luminosity = sol_data.data['luminosity'][0]
    t_eff = sol_data.data['T_eff'][0]
    radius = sol_data.data['radius'][0]
    energy_out = sol_data.data['energy_out'][0]

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


def create_planet_text_html(terra_data, stellar_radius):

    name = terra_data.data['name'][0]
    dist = terra_data.data['xvalues'][0]
    radius = terra_data.data['radius'][0]
    energy_in = terra_data.data['energy_in'][0]

    radius *= AU_IN_M
    dist = (dist - stellar_radius) * AU_IN_M

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