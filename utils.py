"""
Author: Andre Perkins
"""

import pandas

star_color_data = pandas.read_pickle('stellar_color_df.pckl')
AU_IN_M = 149597870700  # meters
LUMINOSITY_OUR_SUN = 3.828e26  # watts


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