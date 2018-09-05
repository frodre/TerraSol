from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div, Slider
from bokeh.layouts import WidgetBox

SUN_COLOR = '#FCD440'
SPACE_COLOR = '#190C26'
IR_COLOR_HOT = '#f43a3a'
IR_COLOR_COLD = '#ff7a1c'
EARTH_COLOR = '#676300'



class EarthEnergy(object):
    """Container for functions and pieces of the earth energy balance web
       application"""

    def __init__(self, frac_cloud=0.7, albedo_cloud=0.4, S0=1.3612e3,
                 frac_land=0.3, albedo_land=0.2, plot_width=800,
                 plot_height=600):

        vis_energy_in = S0 / 4

        self.a_cloud = albedo_cloud
        self.a_land = albedo_land
        self.f_cloud = frac_cloud
        self.f_land = frac_land
        self.alpha = self.calc_albedo()

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
        space_height = plot_height - 150
        vis_to_earth_x = plot_width * 0.25
        vis_to_space_x = plot_width * 0.4
        sun_earth_slope = (earth_top - plot_height) / vis_to_earth_x
        sun_earth_mid_x = vis_to_earth_x / 2
        sun_earth_mid_y = plot_height + sun_earth_slope * sun_earth_mid_x

        earth = p.quad(left=0, right=plot_width,
                       bottom=0, top=earth_top,
                       fill_color=EARTH_COLOR)

        atmos = p.quad(left=0, right=plot_width,
                       bottom=earth_top, top=space_height,
                       fill_color='#ADD8E6', fill_alpha=0.8)

        vis_in = p.line(x=[0, vis_to_earth_x],
                        y=[plot_height, earth_top],
                        color=SUN_COLOR,
                        line_width=20,
                        line_cap='round')

        vis_out = p.line(x=[vis_to_earth_x, vis_to_space_x],
                         y=[earth_top, space_height+50],
                         color=SUN_COLOR,
                         line_width=20,
                         line_cap='round')

        vis_atm_out = p.line(x=[sun_earth_mid_x, vis_to_space_x * 0.5],
                             y=[sun_earth_mid_y, space_height+50],
                             color=SUN_COLOR,
                             line_width=20,
                             line_cap='round')

        vis_atm_trans = p.line(x=[sun_earth_mid_x, vis_to_earth_x],
                               y=[sun_earth_mid_y, earth_top],
                               color=SUN_COLOR,
                               line_width=20, line_cap='round')

        sun = p.circle(x=0.5, y=599.5, fill_color=SUN_COLOR,
                       line_color='#fcb040', radius=80,
                       line_width=3, radius_dimension='x')

        self.plot = p

    def calc_albedo(self):
        cloud = self.f_cloud * self.a_cloud
        land = (1 - self.f_cloud) * self.f_land * self.a_land
        alpha = cloud + land

        return alpha


