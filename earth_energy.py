from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div, Slider
from bokeh.layouts import WidgetBox

import numpy as np
import logging

logger = logging.getLogger(__name__)

SUN_COLOR = '#FCD440'
CLOUD_COLOR = '#e0e0e0'
ATM_COLOR = '#ADD8E6'
SPACE_COLOR = '#190C26'
IR_COLOR_HOT = '#f43a3a'
IR_COLOR_COLD = '#ff7a1c'
EARTH_COLOR = '#676300'
OCN_COLOR = '#3a6fff'

ENERGY_STR = 'energy'
ENERGY_PCT_STR = 'pct_energy'
SFC_NAME = 'Earth_Surface'

ATM_MAX_LAYERS = 10

#Boltzman Constant
SIGMA = 5.67e-8


class EarthEnergy(object):
    """Container for functions and pieces of the earth energy balance web
       application"""

    def __init__(self, frac_cloud=0.7, albedo_cloud=0.4, S0=1.3612e3,
                 frac_land=0.3, albedo_land=0.2, nlayers_atm=1,
                 plot_width=800, plot_height=600):

        self.vis_energy_in = S0 / 4

        self.a_cloud = albedo_cloud
        self.a_land = albedo_land
        self.f_cloud = frac_cloud
        self.f_land = frac_land
        self.total_albedo = self.calc_albedo()
        self.sfc_albedo = self.calc_land_albedo()
        self.atm_albedo = frac_cloud * albedo_cloud
        self.nlayers_atm = nlayers_atm
        self.atm_emissivity = 1.0

        p = figure(x_range=[0, 800], y_range=[0, 600], plot_width=plot_width,
                   plot_height=plot_height)

        # TODO: What's a good title
        p.title.text = "Simple Earth Energy Budget"
        p.background_fill_color = SPACE_COLOR

        # Turn off Axis ticks and labels
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None

        p.xaxis.major_label_text_font_size = '0pt'
        p.yaxis.major_label_text_font_size = '0pt'

        # Turn off grid lines
        p.xgrid.visible = False
        p.ygrid.visible = False

        earth_top = 100
        atm_height = plot_height - 150
        vis_to_earth_x = plot_width * 0.25
        vis_to_space_x = plot_width * 0.4
        vis_to_space_y = atm_height + 50
        sun_earth_slope = (earth_top - plot_height) / vis_to_earth_x
        sun_earth_mid_x = vis_to_earth_x / 2
        sun_earth_mid_y = plot_height + sun_earth_slope * sun_earth_mid_x

        self.atm_y_range = [earth_top, atm_height]

        earth = p.quad(left=0, right=plot_width,
                       bottom=0, top=earth_top,
                       fill_color=EARTH_COLOR)

        self.ocean = p.quad(left=0, right=plot_width,
                            bottom=0, top=earth_top,
                            fill_color=OCN_COLOR, fill_alpha=(1-self.sfc_albedo))

        self.atmos = p.quad(left=0, right=plot_width,
                            bottom=earth_top, top=atm_height,
                            fill_color=ATM_COLOR)

        self.clouds = p.quad(left=0, right=plot_width,
                             bottom=earth_top, top=atm_height,
                             fill_color=CLOUD_COLOR, fill_alpha=self.atm_albedo)

        self.sfc_bnd = None
        self.atm_layer_bnds = None
        self.init_emiss_layers(p, plot_width, earth_top)

        vis_space_to_earth = self._create_solar_ray([0, vis_to_earth_x], [plot_height, earth_top],
                                                    p, 1, 'space_to_earth')
        vis_space_to_atm = self._create_solar_ray([0, sun_earth_mid_x], [plot_height, sun_earth_mid_y],
                                                  p, 1, 'space_to_atm')

        vis_atm_transmit, vis_atm_reflect = self._solar_ray_transmit_reflect(1, self.atm_albedo)

        vis_atm_to_earth = self._create_solar_ray([sun_earth_mid_x, vis_to_earth_x],
                                                  [sun_earth_mid_y, earth_top],
                                                  p, vis_atm_transmit, 'atm_to_earth')
        vis_atm_to_space = self._create_solar_ray([sun_earth_mid_x, vis_to_space_x*0.5],
                                                  [sun_earth_mid_y, vis_to_space_y],
                                                  p, vis_atm_reflect, 'atm_to_space')

        if nlayers_atm >= 1:
            vis_earth_in, vis_earth_reflect = self._solar_ray_transmit_reflect(vis_atm_transmit,
                                                                               self.sfc_albedo)
            vis_space_to_earth.visible = False
        else:
            vis_earth_in, vis_earth_reflect = self._solar_ray_transmit_reflect(1, self.sfc_albedo)
            vis_atm_to_earth.visible = False
            vis_atm_to_space.visible = False
            self.atmos.visible = False

        self.sfc_absorbed_vis = vis_earth_in * self.vis_energy_in

        vis_earth_to_space = self._create_solar_ray([vis_to_earth_x, vis_to_space_x],
                                                    [earth_top, vis_to_space_y],
                                                    p, vis_earth_reflect, 'earth_to_space')

        self.layer_coeffs = None
        self._update_ir_emiss_layers()

        sun = p.circle(x=0.5, y=599.5, fill_color=SUN_COLOR,
                       line_color='#fcb040', radius=80,
                       line_width=3, radius_dimension='x')

        self.direct_rays = {'down': vis_space_to_earth,
                            'up': vis_earth_to_space,
                            'down_atm': vis_space_to_atm}
        self.atm_rays = {'down': vis_atm_to_earth,
                         'up': vis_atm_to_space}

        self.plot = p
        self.albedo_wx = self.init_climate_wx()

    def calc_albedo(self):
        cloud = self.f_cloud * self.a_cloud
        land = (1 - self.f_cloud) * self.f_land * self.a_land
        alpha = cloud + land
        logger.debug(f'Total Albedo Update: {alpha:1.2f}')

        return alpha

    def calc_land_albedo(self):
        sfc_albedo = self.f_land * self.a_land
        logger.debug(f'Total land albedo update: {sfc_albedo:1.2f}')
        return sfc_albedo

    def calc_atm_albedo(self):
        atm_albedo = self.f_cloud * self.a_cloud
        logger.debug(f'Total atm albedo update: {atm_albedo:1.2f}')
        return atm_albedo

    def init_emiss_layers(self, fig_handle, plot_width, earth_top_y):

        self.sfc_bnd = self._create_emiss_bnd(fig_handle,
                                              [0, plot_width],
                                              earth_top_y, '#000000',
                                              SFC_NAME)

        self.atm_layer_bnds = []
        for i in range(ATM_MAX_LAYERS):
            atm_name = 'Atm_Layer_{:d}'.format(i+1)
            layer = self._create_emiss_bnd(fig_handle,
                                           [0, plot_width],
                                           0, '#FFFFFF',
                                           atm_name)
            layer.visible = False
            self.atm_layer_bnds.append(layer)

    def init_climate_wx(self):

        atm_layer_slider = Slider(start=0, end=10, step=1,
                                  value=self.nlayers_atm,
                                  title='Atmosphere: Number of Layers')
        atm_emiss_slider = Slider(start=0, end=1, step=0.05,
                                  value=self.atm_emissivity,
                                  title='Atmosphere Emissivity')

        cloud_frac_slider = Slider(start=0, end=1, step=0.05,
                                   value=self.f_cloud,
                                   title='Cloud Fraction')
        cloud_albedo_slider = Slider(start=0, end=1, step=0.05,
                                     value=self.a_cloud,
                                     title='Cloud Albedo')
        land_frac_slider = Slider(start=0, end=1, step=0.05,
                                  value=self.f_land,
                                  title='Land Fraction')
        land_albedo_slider = Slider(start=0, end=1, step=0.05,
                                    value=self.a_land,
                                    title='Land Albedo')

        # tau_star_opts = [('Mars', '0.125'),
        #                  ('Earth (100 ppm CO2)', '0.66'),
        #                  ('Earth (200 ppm CO2)', '0.75'),
        #                  ('Earth (400 ppm CO2)', '0.84'),
        #                  ('Earth (800 ppm CO2)', '0.93'),
        #                  ('Earth (1600 ppm CO2)', '1.02'),
        #                  ('Earth (3200 ppm CO2)', '1.12'),
        #                  ('Titan', '3'),
        #                  ('Venus', '125')]
        #
        # greenhouse_dropdown = Dropdown(label='Preset Greenhouse Effect',
        #                                button_type='primary',
        #                                menu=tau_star_opts)

        # tau_star_slider = Slider(start=-1, end=np.log10(150), step=0.1,
        #                          value=self.tau_star,
        #                          title='Atmosphere Greenhouse Effect (10^x)')

        def _land_alb_handler(attr, old, new):
            self.a_land = new
            self.alpha = self.calc_albedo()
            self.sfc_albedo = self.calc_land_albedo()
            self._update_land_refl()

        def _land_frac_handler(attr, old, new):
            self.f_land = new
            self.sfc_albedo = self.calc_land_albedo()
            self.alpha = self.calc_albedo()
            self._update_land_refl()

        def _cloud_alb_handler(attr, old, new):
            self.a_cloud = new
            self.alpha = self.calc_albedo()
            self.atm_albedo = self.calc_atm_albedo()
            self._update_atm_refl()

        def _cloud_frac_handler(attr, old, new):
            self.f_cloud = new
            self.alpha = self.calc_albedo()
            self.atm_albedo = self.calc_atm_albedo()
            self._update_atm_refl()

        def _atm_layer_handler(attr, old, new):
            self.nlayers_atm = new

            if old == 0:
                self._turn_on_atm()
            elif old > 0 and new == 0:
                self._turn_off_atm()
            else:
                self._update_ir_emiss_layers()

        def _atm_emiss_handler(attr, old, new):

            self.atm_emissivity = new
            self._update_ir_emiss_layers()

        # def _tau_slider_handler(attr, old, new):
        #     self.tau_star = 10**new
        #     self._update_greenhouse_line()

        # def _tau_dropdown_handler(attr, old, new):
        #     slide_value = np.log10(float(new))
        #     tau_star_slider.value = slide_value
        #     _tau_slider_handler(None, None, slide_value)

        atm_layer_slider.on_change('value', _atm_layer_handler)
        atm_emiss_slider.on_change('value', _atm_emiss_handler)
        cloud_albedo_slider.on_change('value', _cloud_alb_handler)
        cloud_frac_slider.on_change('value', _cloud_frac_handler)
        land_albedo_slider.on_change('value', _land_alb_handler)
        land_frac_slider.on_change('value', _land_frac_handler)
        # tau_star_slider.on_change('value', _tau_slider_handler)
        # greenhouse_dropdown.on_change('value', _tau_dropdown_handler)

        albedo_wx = WidgetBox(land_albedo_slider, land_frac_slider,
                              cloud_albedo_slider, cloud_frac_slider)
        layer_wx = WidgetBox(atm_layer_slider, atm_emiss_slider)
        # tau_wx = WidgetBox(greenhouse_dropdown, tau_star_slider,
        #                    refresh_s0_button)

        # return [albedo_wx, tau_wx]
        return [albedo_wx, layer_wx]

    def _update_land_refl(self):
        if self.nlayers_atm >= 1:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.atm_rays['down'])
        else:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.direct_rays['down'])

        self.ocean.glyph.fill_alpha = (1-self.sfc_albedo)

        sfc_up_data = self.direct_rays['up']

        land_absorb, land_reflect = self._solar_ray_transmit_reflect(in_energy_pct,
                                                                     self.sfc_albedo)

        logger.debug((f'Updating land reflectivity:\n'
                      f'\tIncoming energy = {in_energy:4.1f}\n'
                      f'\tReflected energy = {land_reflect:3.1f}\n'
                      f'\tAbsorbed energy = {land_absorb:3.1f}'))

        self.sfc_absorbed_vis = land_absorb * in_energy

        if in_energy_pct == 0:
            sfc_up_data.visible = False
        else:
            sfc_up_data.visible = True

            up_ray_width = _normalized_ray_width(land_reflect)
            sfc_up_data.glyph.line_width = up_ray_width

        # Update energy absorbed at the surface
        self._update_ray_energy(sfc_up_data, land_reflect)

        # Update the infrared emissions
        self._update_ir_emiss_layers()

    def _update_atm_refl(self):

        [in_pct_energy,
         in_energy] = self._get_ray_energy_and_pct(self.direct_rays['down_atm'])

        self.clouds.glyph.fill_alpha = self.atm_albedo

        [atm_transmit,
         atm_reflect] = self._solar_ray_transmit_reflect(in_pct_energy,
                                                         self.atm_albedo)

        logger.debug((f'Updating atmosphere reflectivity:\n'
                      f'\tIncoming energy = {in_energy:4.1f}\n'
                      f'\tReflected energy = {atm_reflect:3.1f}\n'
                      f'\tTransmitted energy = {atm_transmit:3.1f}'))

        atm_down_ray = self.atm_rays['down']
        atm_up_ray = self.atm_rays['up']
        if atm_transmit == 0:
            atm_down_ray.visible = False
        else:
            atm_down_ray.visible = True
            down_ray_width = _normalized_ray_width(atm_transmit)
            atm_down_ray.glyph.line_width = down_ray_width

        self._update_ray_energy(atm_down_ray, atm_transmit)

        if atm_reflect == 0:
            atm_up_ray.visible = False
        else:
            atm_up_ray.visible = True
            up_ray_width = _normalized_ray_width(atm_reflect)
            atm_up_ray.glyph.line_width = up_ray_width

        self._update_ray_energy(atm_up_ray, atm_reflect)

        self._update_land_refl()

    def _turn_off_atm(self):
        self.atmos.visible = False
        self.clouds.visible = False
        self.atm_albedo = 0
        self._update_atm_refl()

    def _turn_on_atm(self):
        self.atmos.visible = True
        self.clouds.visible = True
        self.atm_albedo = self.calc_atm_albedo()
        self._update_atm_refl()

    def _create_solar_ray(self, x_vals, y_vals, fig_handle, pct_original_in,
                          ray_name):

        tot_energy = self.vis_energy_in * pct_original_in

        data_dict = {'x_vals': x_vals, 'y_vals': y_vals,
                     ENERGY_PCT_STR: [pct_original_in, None],
                     ENERGY_STR: [tot_energy, None]}

        data_src = ColumnDataSource(data=data_dict)

        ray_width = _normalized_ray_width(pct_original_in)

        line = fig_handle.line(x='x_vals', y='y_vals', color=SUN_COLOR,
                               line_width=ray_width, line_cap='round',
                               source=data_src, name=ray_name)

        return line

    def _solve_atm_energy(self):

        in_energy = self.sfc_absorbed_vis
        atm_emiss = self.atm_emissivity
        land_emiss = 1

        coef_matr = []
        for i in range(self.nlayers_atm + 1):
            coef_row = []
            if i == 0:
                absorp_e = land_emiss
            else:
                absorp_e = atm_emiss

            for j in range(self.nlayers_atm + 1):
                layer_loc = i
                dist_from_loc = abs(j - i)

                if j == 0:
                    emiss_e = land_emiss
                else:
                    emiss_e = atm_emiss

                if dist_from_loc == 0:
                    if i == 0:
                        # Surface emits in one direction
                        coef_row.append(1.0)
                    else:
                        coef_row.append(2.0 * emiss_e)
                if dist_from_loc == 1:
                    coef_row.append(-1 * emiss_e * absorp_e)
                elif dist_from_loc >= 2:
                    transmitted = (1 - atm_emiss)**(dist_from_loc - 1)
                    coef = - transmitted * absorp_e * emiss_e
                    coef_row.append(coef)

            coef_matr.append(coef_row)
            logger.debug(f'Row {i} coeffs: {coef_row}')

        A = np.array(coef_matr)

        # A = np.diag([2]*(self.nlayers_atm+1))
        # A = A.astype(np.float)
        #
        # # Fix surface emissivity to one and single output direction
        # A[0, 0] = 1
        #
        # # Direct emissions from other layers
        # if self.nlayers_atm > 0:
        #     A += np.diag([-1 * atm_emiss] * self.nlayers_atm, -1)
        #     A += np.diag([-1 * atm_emiss] * self.nlayers_atm, 1)
        #
        # # Emissions through other layers
        # if self.nlayers_atm >= 2:
        #     for i in range(2, self.nlayers_atm+1):
        #         list_len = self.nlayers_atm - (i - 1)
        #         emiss_term = -(1 - atm_emiss)**(i - 1)
        #         A += np.diag([emiss_term] * list_len, -i)
        #         A += np.diag([emiss_term] * list_len, i)
        #
        # # Emission Adjustment term
        # A[2:, 0] *= atm_emiss
        # A[0, 2:] *= atm_emiss
        # A[1:, 1:] *= atm_emiss

        b = np.array([in_energy] + [0.0] * self.nlayers_atm)

        layer_energy = np.linalg.solve(A, b)

        sfc_energy = layer_energy[0]
        atm_energy = layer_energy[1:]

        return sfc_energy, atm_energy, A

    def _get_layer_y_loc(self):

        earth_top, atm_height = self.atm_y_range
        y_dist = atm_height - earth_top
        delta_y = y_dist / (self.nlayers_atm + 1)
        y_locs = [earth_top + (i + 1)*delta_y for i in range(self.nlayers_atm)]
        return y_locs

    def _update_ir_emiss_layers(self):

        sfc_energy, atm_energy, layer_coefs = self._solve_atm_energy()

        self._update_emiss_bnd(self.sfc_bnd, sfc_energy)

        atm_y_locs = self._get_layer_y_loc()
        for i in range(self.nlayers_atm):
            emiss_bnd = self.atm_layer_bnds[i]
            emiss_bnd.visible = True
            y_loc = atm_y_locs[i]
            bnd_energy = atm_energy[i]

            self._update_emiss_bnd(emiss_bnd, bnd_energy, y=y_loc)

        for i in range(self.nlayers_atm, ATM_MAX_LAYERS):
            self.atm_layer_bnds[i].visible = False

        # Coefficients for calculating the layer energy
        self.layer_coeffs = layer_coefs

    @staticmethod
    def _update_emiss_bnd(bnd, energy, y=None):

        name = bnd.name
        src = bnd.data_source.data
        temperature = _calc_temp(energy)
        src['layer_energy'][0] = energy
        src['layer_temp'][0] = temperature

        if y is not None:
            src['y_vals'] = [y, y]

        if name != SFC_NAME:
            # Layer emits energy up and down
            print_energy = energy*2
        else:
            print_energy = energy

        logger.debug(f'Update layer emissions, {name}:\n'
                     f'\tTotal Energy Output: {print_energy:4.1f} W/m2'
                     f'\tTemperature: {temperature:3.1f} K')

    @staticmethod
    def _create_emiss_bnd(fig_handle, x_vals, y_val, line_color, layer_name):

        data_dict = {'layer_energy': [0, None],
                     'layer_temp': [273, None],
                     'x_vals': x_vals,
                     'y_vals': [y_val, y_val]}

        layer_src = ColumnDataSource(data=data_dict)

        line = fig_handle.line(x='x_vals', y='y_vals', line_width=2,
                               line_color=line_color, source=layer_src,
                               name=layer_name)

        return line

    def _update_ray_energy(self, ray, pct_initial_energy):

        energy = self.vis_energy_in
        new_energy = energy * pct_initial_energy
        ray.data_source.data[ENERGY_PCT_STR][0] = pct_initial_energy
        ray.data_source.data[ENERGY_STR][0] = new_energy

        logger.debug(f'Updating Ray Energy: {ray.name} -- pct_init: '
                     f'{pct_initial_energy:1.2f} -- energy: {new_energy:3.1f}')

    @staticmethod
    def _get_ray_energy_and_pct(ray):

        pct_energy = ray.data_source.data[ENERGY_PCT_STR][0]
        energy = ray.data_source.data[ENERGY_STR][0]

        return pct_energy, energy

    @staticmethod
    def _solar_ray_transmit_reflect(incoming_pct, albedo):

        pct_reflected = incoming_pct * albedo
        pct_transmitted = incoming_pct - pct_reflected

        return pct_transmitted, pct_reflected


def _normalized_ray_width(pct_transmitted):

    linewidth_max = 20
    linewidth_min = 5

    width_range = linewidth_max - linewidth_min

    width = linewidth_min + width_range * pct_transmitted

    return width

def _calc_temp(energy):

    temp = (energy / SIGMA) ** (1/4)

    return temp




