from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Arrow, OpenHead, NormalHead
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

ATM_MAX_LAYERS = 1
EARTH_TOP = 100

#Boltzman Constant
SIGMA = 5.67e-8

#CSS Styles
title_style = '"text-align:left; font-weight:bold; font-size:20px"'
left_style = '"text-align:left; font-weight:bold; font-size=16px"'
right_style = '"text-align:right; font-weight:normal; font-size=16px"'


class EarthEnergy(object):
    """Container for functions and pieces of the earth energy balance web
       application"""

    def __init__(self, frac_cloud=0.7, albedo_cloud=0.4, s0=1.3612e3,
                 frac_land=0.3, albedo_land=0.2, nlayers_atm=1,
                 plot_width=800, plot_height=600, simple_albedo=True,
                 albedo=0.3):

        self.s0 = s0
        self.vis_energy_in = s0 / 4

        if simple_albedo:
            self.simple_albedo = simple_albedo
            self.a_cloud = 0
            self.f_cloud = 0
            self.f_land = 1
            self.a_land = albedo
        else:
            self.a_cloud = albedo_cloud
            self.a_land = albedo_land
            self.f_cloud = frac_cloud
            self.f_land = frac_land

        self.total_albedo = self.calc_albedo()
        self.sfc_albedo = self.calc_land_albedo()
        self.atm_albedo = self.calc_atm_albedo()
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

        atm_height = plot_height - 150

        self.atm_y_range = [EARTH_TOP, atm_height]
        self.emiss_arrow_range = [plot_width * 0.55, plot_width * 0.95]

        earth = p.quad(left=0, right=plot_width,
                       bottom=0, top=EARTH_TOP,
                       fill_color=EARTH_COLOR)

        # self.ocean = p.quad(left=0, right=plot_width,
        #                     bottom=0, top=EARTH_TOP,
        #                     fill_color=OCN_COLOR, fill_alpha=(1-self.sfc_albedo))

        self.atmos = p.quad(left=0, right=plot_width,
                            bottom=EARTH_TOP, top=atm_height,
                            fill_color=ATM_COLOR)

        self.clouds = p.quad(left=0, right=plot_width,
                             bottom=EARTH_TOP, top=atm_height,
                             fill_color=CLOUD_COLOR, fill_alpha=self.atm_albedo)

        self.sfc_bnd = None
        self.atm_layer_bnds = None

        self.init_emiss_layers(p, plot_width, EARTH_TOP)
        self.layer_arrows = self._create_arrows(p, plot_width)

        self.direct_rays = None
        self.atm_rays = None
        self.init_solar_rays(plot_height, plot_width, atm_height, p)

        # TODO: Temporary for 1-layer atms
        self.sfc_ir = self.layer_arrows[0][0][0]
        self.atm_ir_pass = self.layer_arrows[1][0][0]
        self.atm_ir = self.layer_arrows[1][1]

        self.layer_coeffs = None
        self.info_div = Div(text='', width=325, height=175)

        self._update_atm_refl()

        if nlayers_atm == 0:
            self._turn_off_atm()

        sun = p.circle(x=0.5, y=599.5, fill_color=SUN_COLOR,
                       line_color='#fcb040', radius=80,
                       line_width=3, radius_dimension='x')

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

    def init_solar_rays(self, plot_height, plot_width, atm_height, fig_handle):

        vis_to_earth_x = plot_width * 0.25
        vis_to_space_x = plot_width * 0.4
        vis_to_space_y = atm_height + 50
        sun_earth_slope = (EARTH_TOP - plot_height) / vis_to_earth_x
        sun_earth_mid_x = vis_to_earth_x / 2
        sun_earth_mid_y = plot_height + sun_earth_slope * sun_earth_mid_x

        vis_space_to_earth = self._create_solar_ray([0, vis_to_earth_x],
                                                    [plot_height, EARTH_TOP],
                                                    fig_handle, 'space_to_earth',
                                                    energy_pct=1,
                                                    energy=self.vis_energy_in)
        vis_space_to_atm = self._create_solar_ray([0, sun_earth_mid_x],
                                                  [plot_height,
                                                   sun_earth_mid_y],
                                                  fig_handle, 'space_to_atm',
                                                  energy_pct=1,
                                                  energy=self.vis_energy_in)

        vis_atm_to_earth = self._create_solar_ray(
            [sun_earth_mid_x, vis_to_earth_x],
            [sun_earth_mid_y, EARTH_TOP],
            fig_handle, 'atm_to_earth')
        vis_atm_to_space = self._create_solar_ray(
            [sun_earth_mid_x, vis_to_space_x * 0.5],
            [sun_earth_mid_y, vis_to_space_y],
            fig_handle, 'atm_to_space')

        vis_earth_to_space = self._create_solar_ray(
            [vis_to_earth_x, vis_to_space_x],
            [EARTH_TOP, vis_to_space_y],
            fig_handle, 'earth_to_space')

        self.direct_rays = {'down': vis_space_to_earth,
                            'up': vis_earth_to_space,
                            'down_atm': vis_space_to_atm}
        self.atm_rays = {'down': vis_atm_to_earth,
                         'up': vis_atm_to_space}

    def init_emiss_layers(self, fig_handle, plot_width, earth_top_y):

        self.sfc_bnd = self._create_emiss_bnd(fig_handle,
                                              [0, plot_width],
                                              earth_top_y, '#000000',
                                              SFC_NAME)

        # self.sfc_arrows = self._create_arrows(fig_handle, plot_width)

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

        atm_layer_slider = Slider(start=0, end=1, step=1,
                                  value=self.nlayers_atm,
                                  title='Atmosphere: Number of Layers')
        atm_emiss_slider = Slider(start=0.05, end=1, step=0.05,
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

        simple_albedo_slider = Slider(start=0, end=0.99, step=0.01,
                                      value=self.a_land,
                                      title='Planetary Albedo')

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
            if self.nlayers_atm > 0:
                self._update_atm_refl()

        def _cloud_frac_handler(attr, old, new):
            self.f_cloud = new
            self.alpha = self.calc_albedo()
            self.atm_albedo = self.calc_atm_albedo()
            if self.nlayers_atm > 0:
                self._update_atm_refl()

        def _simple_alb_handler(attr, old, new):
            self.a_land = new
            self.alpha = self.calc_albedo()
            self.sfc_albedo = self.calc_land_albedo()
            self._update_land_refl()

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
        simple_albedo_slider.on_change('value', _simple_alb_handler)
        # tau_star_slider.on_change('value', _tau_slider_handler)
        # greenhouse_dropdown.on_change('value', _tau_dropdown_handler)

        if self.simple_albedo:
            albedo_wx = WidgetBox(simple_albedo_slider)
        else:
            albedo_wx = WidgetBox(land_albedo_slider, land_frac_slider,
                                  cloud_albedo_slider, cloud_frac_slider)

        layer_wx = WidgetBox(atm_layer_slider, atm_emiss_slider)
        # tau_wx = WidgetBox(greenhouse_dropdown, tau_star_slider,
        #                    refresh_s0_button)

        # return [albedo_wx, tau_wx]
        return [albedo_wx, layer_wx]

    def _update_land_refl(self):
        if self.nlayers_atm >= 1 and not self.simple_albedo:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.atm_rays['down'])
        else:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.direct_rays['down'])

        # self.ocean.glyph.fill_alpha = (1-self.sfc_albedo)

        sfc_up_data = self.direct_rays['up']

        land_absorb, land_reflect = self._solar_ray_transmit_reflect(in_energy_pct,
                                                                     self.sfc_albedo)

        self.sfc_absorbed_vis = land_absorb * in_energy

        logger.debug((f'Updating land reflectivity:\n'
                      f'\tIncoming energy = {in_energy:4.1f}\n'
                      f'\tReflected energy pct = {land_reflect:3.1f}\n'
                      f'\tAbsorbed energy pct = {land_absorb:3.1f}\n'
                      f'\tTot Absorb energy = {self.sfc_absorbed_vis:3.1f}'))

        if in_energy_pct == 0 or land_reflect == 0:
            sfc_up_data.visible = False
        else:
            sfc_up_data.visible = True

            up_ray_width = _normalized_line_width(land_reflect)
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
            down_ray_width = _normalized_line_width(atm_transmit)
            atm_down_ray.glyph.line_width = down_ray_width

        self._update_ray_energy(atm_down_ray, atm_transmit)

        if atm_reflect == 0:
            atm_up_ray.visible = False
        else:
            atm_up_ray.visible = True
            up_ray_width = _normalized_line_width(atm_reflect)
            atm_up_ray.glyph.line_width = up_ray_width

        self._update_ray_energy(atm_up_ray, atm_reflect)

        self._update_land_refl()

    def _turn_off_atm(self):
        self.atmos.visible = False
        self.clouds.visible = False
        self.atm_albedo = 0
        self._update_atm_refl()
        self._turn_off_atm_arrows()

    def _turn_on_atm(self):
        self.atmos.visible = True
        self.clouds.visible = True
        self.atm_albedo = self.calc_atm_albedo()
        self._update_atm_refl()
        self._turn_on_atm_arrows()

    @staticmethod
    def _create_solar_ray(x_vals, y_vals, fig_handle, ray_name,
                          energy_pct=None, energy=None):

        data_dict = {'x_vals': x_vals, 'y_vals': y_vals,
                     ENERGY_PCT_STR: [energy_pct, None],
                     ENERGY_STR: [energy, None]}

        data_src = ColumnDataSource(data=data_dict)

        if energy_pct is None:
            ray_width = 5
        else:
            ray_width = _normalized_line_width(energy_pct)

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

        A[abs(A) < 1e-5] = 0

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

        if atm_y_locs:
            delta_y = atm_y_locs[0] - self.atm_y_range[0]
        else:
            delta_y = np.diff(self.atm_y_range)[0]

        self.sfc_ir.visible = True
        sfc_src = self.sfc_ir.source
        new_y_vals = {'y_start': [self.atm_y_range[0]],
                      'y_end': [self.atm_y_range[0] + delta_y * 0.66]}
        new_locs = self._new_arrow_dict(sfc_src.data,
                                        new_y_vals)
        sfc_src.data = new_locs

        linewidth = _normalized_line_width(1)
        self.sfc_ir.line_width = linewidth
        self.sfc_ir.end.size = linewidth * 1.5
        self.sfc_ir.end.line_width = linewidth * 0.25

        for i in range(self.nlayers_atm):
            emiss_bnd = self.atm_layer_bnds[i]
            emiss_bnd.visible = True
            y_loc = atm_y_locs[i]
            bnd_energy = atm_energy[i]

            self._update_emiss_bnd(emiss_bnd, bnd_energy, y=y_loc)
            self.handle_1_layer(bnd_energy, sfc_energy, y_loc,
                                self.emiss_arrow_range[0],
                                delta_y)

        for i in range(self.nlayers_atm, ATM_MAX_LAYERS):
            self.atm_layer_bnds[i].visible = False

        # Coefficients for calculating the layer energy
        self.layer_coeffs = layer_coefs

        self.info_div.text = self.create_model_summary_text()

        # self._update_ir_arrows()

    @staticmethod
    def _new_arrow_dict(old_dict, new_kv_pairs):

        new_dict = {}
        for k, v in old_dict.items():
            if k in new_kv_pairs:
                new_dict[k] = new_kv_pairs[k]
            else:
                new_dict[k] = v

        return new_dict

    def _update_ir_arrow_locs(self, layer_idx, layer_arrows, y, delta_y):

        delta_x = np.diff(self.emiss_arrow_range)[0] / (self.nlayers_atm + 1)
        x0 = self.emiss_arrow_range[0]
        x_locs = [x0 + delta_x * i for i in range(self.nlayers_atm + 1)]

        iter_dist = min(len(layer_arrows), self.nlayers_atm + 1)

        for i in range(iter_dist):

            arrow_grp = layer_arrows[i]

            for j, arrow in enumerate(arrow_grp):
                arrow.visible = True
                arrow.x_start = x_locs[i]
                arrow.x_end = x_locs[i]

                if i == layer_idx:
                    if j >= 1:
                        y_end = y - delta_y*0.66
                    else:
                        y_end = y + delta_y*0.66
                elif i < layer_idx:
                    y_end = y + delta_y*0.66
                else:
                    y_end = y - delta_y*0.66

                arrow.y_start = y
                arrow.y_end = y_end

        for i in range(iter_dist, self.nlayers_atm + 1):
            try:
                arrow_grp = layer_arrows[i]
                for arrow in arrow_grp:
                    arrow.visible = False
            except IndexError:
                break

    def _update_ir_arrows(self):

        emiss_coefs = self.layer_coeffs
        sfc_energy = [self.sfc_bnd.data_source.data['layer_energy'][0]]
        atm_energy = [bnd.data_source.data['layer_energy'][0]
                      for bnd in self.atm_layer_bnds]
        layer_energy = sfc_energy + atm_energy

        for i in range(self.nlayers_atm + 1):

            curr_coefs = emiss_coefs[i]
            arrows = self.layer_arrows[i]

            for j, arrow_grp in enumerate(arrows):
                coef = curr_coefs[j]
                curr_energy = layer_energy[j]

                if j == i:
                    energy_pct = (abs(coef) * curr_energy) / sfc_energy[0]
                elif j < i:
                    energy_pct = (sfc_energy[0] * (1-abs(coef))) / sfc_energy[0]
                else:
                    # TODO: Figure this out
                    energy_pct = None

                linewidth = _normalized_line_width(energy_pct)

                for arrow in arrow_grp:
                    if energy_pct == 0:
                        arrow.visible = False
                    else:
                        arrow.visible = True
                        arrow.line_width = linewidth
                        arrow.end.size = linewidth*1.25

    def handle_1_layer(self, bnd_energy, sfc_energy, y_loc, x_loc, delta_y):

        coef = self.atm_emissivity
        pass_coef = (1 - coef)

        logger.debug(f'New pass: {pass_coef:1.2f}')

        linewidth_pass = _normalized_line_width(pass_coef)
        linewidth = _normalized_line_width(coef*bnd_energy / self.sfc_absorbed_vis)

        if pass_coef == 0:
            self.atm_ir_pass.visible = False
        else:
            logger.debug(f'pass is not zero: linewidth={linewidth_pass}')
            self.atm_ir_pass.line_width = linewidth_pass
            self.atm_ir_pass.end.size = linewidth_pass * 1.5
            self.atm_ir_pass.end.line_width = linewidth * 0.2
            new_locs = {'x_start': [x_loc],
                        'x_end': [x_loc],
                        'y_start': [y_loc],
                        'y_end': [y_loc + delta_y * 0.66]}
            data_dict = self._new_arrow_dict(self.atm_ir_pass.source.data,
                                             new_locs)
            self.atm_ir_pass.source.data = data_dict
            self.atm_ir_pass.visible = True

        for i in range(len(self.atm_ir)):
            emit = self.atm_ir[i]

            if i != 0:
                delta_y = -delta_y

            emit.visible = True
            emit.line_width = linewidth
            emit.end.size = linewidth * 1.5
            emit.end.line_width = linewidth * 0.2
            new_locs = {'x_start': [x_loc + 100],
                        'x_end': [x_loc + 100],
                        'y_start': [y_loc],
                        'y_end': [y_loc + delta_y * 0.66]}
            data_dict = self._new_arrow_dict(emit.source.data,
                                             new_locs)
            emit.source.data = data_dict

    def _turn_off_atm_arrows(self):

        self.atm_ir[0].visible = False
        self.atm_ir[1].visible = False
        self.atm_ir_pass.visible = False

    def _turn_on_atm_arrows(self):

        self.atm_ir[0].visible = True
        self.atm_ir[1].visible = True

        if self.atm_emissivity == 1:
            self.atm_ir_pass.visible = False
        else:
            self.atm_ir_pass.visible = True

    def create_model_summary_text(self):

        sfc_temp = self.sfc_bnd.data_source.data['layer_temp'][0]
        sfc_temp -= 273
        sfc_temp_f = sfc_temp * 9/5 + 32

        if self.nlayers_atm == 1:
            atm_temp = self.atm_layer_bnds[0].data_source.data['layer_temp'][0]
            atm_temp -= 273
            atm_temp_f = atm_temp * 9 / 5 + 32
            logger.debug(('Atm temp C: {:2.1f} Atm temp F: {:3.1f}'
                          ''.format(atm_temp, atm_temp_f)))
        else:
            atm_temp_f = None

        tot_albedo = self.total_albedo
        if atm_temp_f is not None:

            atm_temp_text = """
            <tr>
                <td style={left_style}>Atmos. Layer Temperature</td>
                <td style={right_style}> {atm_temp_f:3.1f} F</td>
            </tr>
            """.format(left_style=left_style, right_style=right_style,
                       atm_temp_f=atm_temp_f)
        else:
            atm_temp_text = ''

        solar_constant = self.s0
        incident_energy = self.vis_energy_in

        text = """
        <table style="width:100%">
            <tr><th colspan="2" style={title_style}>Energy Budget Characteristics</th></tr>
            <tr>
                <td style={left_style}>Solar Constant: </td>
                <td style={right_style}>{solar_constant:4.2f} W/m^2</td>
            </tr>
            <tr>
                <td style={left_style}>Earth Incident Energy:</td>
                <td style={right_style}> {incident_energy:4.2f} W/m^2</td>
            </tr>
            <tr>
                <td style={left_style}>Planetary Albedo: </td>
                <td style={right_style}>{albedo:1.2f}</td>
            </tr>
            <tr>
                <td style={left_style}>Surface Temperature:</td>
                <td style={right_style}>{sfc_temp:3.1f} F</td>
            </tr>
            {atm_temp_text}
        </table>""".format(title_style=title_style,
                           left_style=left_style,
                           right_style=right_style,
                           solar_constant=solar_constant,
                           incident_energy=incident_energy,
                           albedo=tot_albedo,
                           sfc_temp=sfc_temp_f,
                           atm_temp_text=atm_temp_text)

        return text

    @staticmethod
    def _create_arrows(fig_handle, plot_width):

        x0 = plot_width * 0.55

        all_arrows = []
        for i in range(ATM_MAX_LAYERS + 1):
            layer_arrows = []

            for j in range(ATM_MAX_LAYERS + 1):

                if i == j:
                    head = NormalHead(fill_color=IR_COLOR_HOT)
                    head2 = NormalHead(fill_color=IR_COLOR_HOT)
                    l_alpha = 1.0
                else:
                    l_alpha = 0.8
                    head = OpenHead(line_color=IR_COLOR_HOT,
                                    line_alpha=l_alpha)
                    head2 = OpenHead(line_color=IR_COLOR_HOT,
                                     line_alpha=l_alpha)

                data_src = ColumnDataSource(data=dict(x_start=[x0],
                                                      x_end=[x0],
                                                      y_start=[0],
                                                      y_end=[1]))
                data_src2 = ColumnDataSource(data=dict(x_start=[x0],
                                                       x_end=[x0],
                                                       y_start=[0],
                                                       y_end=[1]))

                up_arrow = Arrow(end=head, x_start='x_start', x_end='x_end',
                                 y_start='y_start', y_end='y_end',
                                 line_color=IR_COLOR_HOT,
                                 line_alpha=l_alpha,
                                 source=data_src)

                down_arrow = Arrow(end=head2, x_start='x_start', x_end='x_end',
                                   y_start='y_start', y_end='y_end',
                                   line_color=IR_COLOR_HOT,
                                   line_alpha=l_alpha,
                                   source=data_src2)

                up_arrow.visible = False
                down_arrow.visible = False

                if j < i:
                    fig_handle.add_layout(up_arrow)
                    layer_arrows.append((up_arrow,))
                elif j == i:
                    fig_handle.add_layout(up_arrow)

                    if i == 0:
                        layer_arrows.append((up_arrow,))
                    else:
                        fig_handle.add_layout(down_arrow)
                        layer_arrows.append((up_arrow, down_arrow))
                else:
                    if i != 0:
                        fig_handle.add_layout(down_arrow)
                        layer_arrows.append((down_arrow,))

            all_arrows.append(layer_arrows)

        return all_arrows

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


def _normalized_line_width(pct_transmitted, linewidth_max=20, linewidth_min=5):

    width_range = linewidth_max - linewidth_min

    width = linewidth_min + width_range * pct_transmitted

    return width


def _calc_temp(energy):

    temp = (energy / SIGMA) ** (1/4)

    return temp




