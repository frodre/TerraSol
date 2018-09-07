from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div, Slider
from bokeh.layouts import WidgetBox

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

        earth = p.quad(left=0, right=plot_width,
                       bottom=0, top=earth_top,
                       fill_color=EARTH_COLOR)

        self.ocean = p.quad(left=0, right=plot_width,
                            bottom=0, top=earth_top,
                            fill_color=OCN_COLOR, fill_alpha=(1-self.sfc_albedo))

        atmos = p.quad(left=0, right=plot_width,
                       bottom=earth_top, top=atm_height,
                       fill_color=ATM_COLOR)

        self.clouds = p.quad(left=0, right=plot_width,
                             bottom=earth_top, top=atm_height,
                             fill_color=CLOUD_COLOR, fill_alpha=self.atm_albedo)

        vis_space_to_earth = self._create_solar_ray([0, vis_to_earth_x], [plot_height, earth_top],
                                                    p, 1)
        vis_space_to_atm = self._create_solar_ray([0, sun_earth_mid_x], [plot_height, sun_earth_mid_y],
                                                  p, 1)

        vis_atm_transmit, vis_atm_reflect = self._solar_ray_transmit_reflect(1, self.atm_albedo)

        vis_atm_to_earth = self._create_solar_ray([sun_earth_mid_x, vis_to_earth_x],
                                                  [sun_earth_mid_y, earth_top],
                                                  p, vis_atm_transmit)
        vis_atm_to_space = self._create_solar_ray([sun_earth_mid_x, vis_to_space_x*0.5],
                                                  [sun_earth_mid_y, vis_to_space_y],
                                                  p, vis_atm_reflect)

        if nlayers_atm >= 1:
            _, vis_earth_reflect = self._solar_ray_transmit_reflect(vis_atm_transmit, self.sfc_albedo)
            vis_space_to_earth.visible = False
        else:
            _, vis_earth_reflect = self._solar_ray_transmit_reflect(1, self.sfc_albedo)
            vis_atm_to_earth.visible = False
            vis_atm_to_space.visible = False

        vis_earth_to_space = self._create_solar_ray([vis_to_earth_x, vis_to_space_x],
                                                    [earth_top, vis_to_space_y],
                                                    p, vis_earth_reflect)

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

        return alpha

    def calc_land_albedo(self):
        sfc_albedo = self.f_land * self.a_land
        return sfc_albedo

    def calc_atm_albedo(self):
        atm_albedo = self.f_cloud * self.a_cloud
        return atm_albedo

    def init_climate_wx(self):

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

        # def _tau_slider_handler(attr, old, new):
        #     self.tau_star = 10**new
        #     self._update_greenhouse_line()

        # def _tau_dropdown_handler(attr, old, new):
        #     slide_value = np.log10(float(new))
        #     tau_star_slider.value = slide_value
        #     _tau_slider_handler(None, None, slide_value)

        cloud_albedo_slider.on_change('value', _cloud_alb_handler)
        cloud_frac_slider.on_change('value', _cloud_frac_handler)
        land_albedo_slider.on_change('value', _land_alb_handler)
        land_frac_slider.on_change('value', _land_frac_handler)
        # tau_star_slider.on_change('value', _tau_slider_handler)
        # greenhouse_dropdown.on_change('value', _tau_dropdown_handler)

        albedo_wx = WidgetBox(land_albedo_slider, land_frac_slider,
                              cloud_albedo_slider, cloud_frac_slider)
        # tau_wx = WidgetBox(greenhouse_dropdown, tau_star_slider,
        #                    refresh_s0_button)

        # return [albedo_wx, tau_wx]
        return [albedo_wx]

    def _update_land_refl(self):
        if self.nlayers_atm >= 1:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.atm_rays['down'])
        else:
            in_energy_pct, in_energy = self._get_ray_energy_and_pct(self.direct_rays['down'])

        self.ocean.glyph.fill_alpha = (1-self.sfc_albedo)

        sfc_up_data = self.direct_rays['up']

        _, land_reflect = self._solar_ray_transmit_reflect(in_energy_pct, self.sfc_albedo)
        if in_energy_pct == 0:
            sfc_up_data.visible = False
        else:
            sfc_up_data.visible = True

            up_ray_width = _normalized_ray_width(land_reflect)
            sfc_up_data.glyph.line_width = up_ray_width

        self._update_ray_energy(sfc_up_data, land_reflect, in_energy)

    def _update_atm_refl(self):

        [in_pct_energy,
         in_energy] = self._get_ray_energy_and_pct(self.direct_rays['down_atm'])

        self.clouds.glyph.fill_alpha = self.atm_albedo

        [atm_transmit,
         atm_reflect] = self._solar_ray_transmit_reflect(in_pct_energy,
                                                         self.atm_albedo)

        atm_down_ray = self.atm_rays['down']
        atm_up_ray = self.atm_rays['up']

        if atm_transmit == 0:
            atm_down_ray.visible = False
        else:
            atm_down_ray.visible = True
            down_ray_width = _normalized_ray_width(atm_transmit)
            atm_down_ray.glyph.line_width = down_ray_width

        self._update_ray_energy(atm_down_ray, atm_transmit, in_energy)

        if atm_reflect == 0:
            atm_up_ray.visible = False
        else:
            atm_up_ray.visible = True
            up_ray_width = _normalized_ray_width(atm_reflect)
            atm_up_ray.glyph.line_width = up_ray_width

        self._update_ray_energy(atm_up_ray, atm_reflect, in_energy)

        self._update_land_refl()

    def _create_solar_ray(self, x_vals, y_vals, fig_handle, pct_original_in):

        tot_energy = self.vis_energy_in * pct_original_in

        data_dict = {'x_vals': x_vals, 'y_vals': y_vals,
                     ENERGY_PCT_STR: [pct_original_in, None],
                     ENERGY_STR: [tot_energy, None]}

        data_src = ColumnDataSource(data=data_dict)

        ray_width = _normalized_ray_width(pct_original_in)

        line = fig_handle.line(x='x_vals', y='y_vals', color=SUN_COLOR,
                               line_width=ray_width, line_cap='round',
                               source=data_src)

        return line

    @staticmethod
    def _update_ray_energy(ray, pct_initial_energy, energy):

        ray.data_source.data[ENERGY_PCT_STR][0] = pct_initial_energy
        ray.data_source.data[ENERGY_STR][0] = energy * pct_initial_energy

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




