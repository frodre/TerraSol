# Climate model based off of Judy and Hansi's energy balance notebook
from bokeh.models import ColumnDataSource, WidgetBox
from bokeh.models.widgets import TextInput, Select, Slider, Button
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import xarray as xr


class PlanetClimateEBM(object):

    """
    Planetary Climate class that uses the North energy balance model.
    """

    def __init__(self, terra_sol_obj, plot_width=800):

        energy_in = terra_sol_obj.get_planet_energy_in()
        self._terra_sol = terra_sol_obj

        planet_energy_in = energy_in / 4
        self._plot_width = plot_width

        climate_model_inputs = ColumnDataSource(
            data=dict(A=[211.22], B=[2.1], Q=[planet_energy_in],
                      D=[1.2], S2=[-0.482], C=[9.8], nlats=[70],
                      tol=[1e-5], init_condition=['normal']))
        self.model_input = climate_model_inputs

        # Initialize EnergyBalanceModel and solve initial climate
        self.planet_climate = None
        self.climate_result = None
        self.final_T_dataframe = None
        self._update_model_and_run_sim()

        # Create user inputs for EBM parameters
        (self.calc_button,
         self.input_wx,
         (self.float_inputtext,
          self.general_input)) = self.init_climate_input_wx(self.planet_climate)

    def update_planet_climate(self):
        self.calc_button.disabled = True
        valid_float_in = {key: [float(field.value.strip())]
                          for key, field in self.float_inputtext.items()}
        general_in = {key: [field.value]
                      for key, field in self.general_input.items()}

        self.model_input.data.update(**valid_float_in)
        self.model_input.data.update(**general_in)

        self._update_model_and_run_sim()
        self.calc_button.disabled = False

    def _update_model_and_run_sim(self):
        new_kwargs = self.get_ebm_kwargs()
        self.planet_climate = EnergyBalanceModel(**new_kwargs)

        self.climate_result = self.planet_climate.solve_climate()
        self.final_T_dataframe = self.planet_climate.convert_1d_to_grid()

        print(self.climate_result)

    def _update_energy_in(self):
        energy_in = self._terra_sol.get_planet_energy_in()
        planet_energy_in = energy_in / 4

        self.model_input.data.update(dict(Q=[planet_energy_in]))
        energy_str = '{:.2f}'.format(planet_energy_in)
        self.float_inputtext['Q'].value = energy_str

    def init_climate_input_wx(self, planet_climate):
        # Climate inputs
        planet_emiss = TextInput(title='Planetary IR energy out (W/m^2)',
                                 value='{:.2f}'.format(planet_climate.A))
        planet_atm_forcing = TextInput(title='Atmosphere IR adjustment (W/m^2)',
                                       value='{:.1f}'.format(planet_climate.B))
        solar_input = TextInput(title='Incoming solar (W/m^2) [Divided by 4]',
                                value='{:.2f}'.format(planet_climate.Q))
        energy_transport = TextInput(
            title='Energy transport towards poles (1/C)',
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
        calc_climate = Button(label='Simulate Climate', button_type='success')
        calc_climate.on_click(self.update_planet_climate)
        refresh_energy_in = Button(label='Refresh Solar Input')
        refresh_energy_in.on_click(self._update_energy_in)

        float_input = {'A': planet_emiss, 'B': planet_atm_forcing,
                        'Q': solar_input, 'D': energy_transport,
                        'S2': s2_input, 'C': heat_capacity}
        general_input = {'nlats': numlats,
                         'init_condition': init_planet_T}

        clim_input_grp1 = WidgetBox(children=[planet_emiss,
                                              planet_atm_forcing,
                                              solar_input,
                                              refresh_energy_in],
                                    width=int(self._plot_width/3))
        clim_input_grp2 = WidgetBox(energy_transport, s2_input, heat_capacity)
        clim_input_grp3 = WidgetBox(numlats, init_planet_T)

        return (calc_climate,
                [clim_input_grp1, clim_input_grp2, clim_input_grp3],
                (float_input, general_input))

    def get_ebm_kwargs(self):

        ebm_kwargs = {key: model_in[0] for key, model_in in
                      self.model_input.data.items()}
        return ebm_kwargs






def calc_albedo(x, temperature):
    a_ice = 0.6
    a_noice_ocn = 0.263
    a_noice_cloud = 0.04  # why is this so low?

    albedo = a_ice * np.ones(len(x))
    tcondition = temperature >= -2
    albedo[tcondition] = a_noice_ocn + a_noice_cloud * (3*x[tcondition]**2 - 1)

    return albedo


def start_conditions(nlats, condition):
    temp_0 = np.ones(nlats)

    if condition == 'cold':
        temp_0 *= -10
    elif condition == 'normal':
        temp_0 *= 5
    elif condition == 'warm':
        temp_0 *= 15
    else:
        raise ValueError('Unrecognized initial condition: {}'.format(condition))

    return temp_0


class EnergyBalanceModel(object):

    ITERMAX = 10000

    def __init__(self, A=211.22, B=2.1, Q=338.52, D=1.2, S2=-0.482, C=9.8,
                       nlats=70, tol=1e-5, init_condition='normal'):
        self.A = A
        self.B = B
        self.Q = Q
        self.D = D
        self.S2 = S2
        self.C = C
        self.nlats = nlats
        self.tol = tol
        self.init_condition = init_condition

        self.x = np.linspace(-1, 1, nlats)
        self.lats = np.arcsin(self.x)
        self.lats_in_deg = self.lats * 180 / np.pi
        self.lons_in_deg = np.arange(0, 360, 5)

        self._dx = self.x[1] - self.x[0]
        self._dt = self._dx**2 / D

        self.s = Q * (1 + (S2 * (3 * self.x**2 - 1) / 2))
        self.T = start_conditions(nlats, init_condition)
        self.T_mean = self.calc_mean_T()
        self.freeze_lat = self.calc_iceline()

        self.iter_T_means = None
        self.iter_freeze_lat = None

        self.eps = 1
        self.niters = 0

    def solve_climate(self):

        result = ""
        self.iter_T_means = []
        self.iter_freeze_lat = []

        dx = self._dx
        dt = self._dt

        term1 = self.D * (1 - self.x**2) / dx**2
        term2 = self.D * self.x / (2 * dx)

        for curr_iter in range(self.ITERMAX):

            alpha = calc_albedo(self.x, self.T)
            insolation = self.s * (1 - alpha)

            # Set up tridiagonal matrix
            T_p1 = np.roll(self.T, -1)
            T_m1 = np.roll(self.T, 1)

            diag = (self.C / self._dt) + term1 + (self.B / 2)
            lodiag = -term1 / 2 - term2
            updiag = -term1 / 2 + term2

            mydata = [lodiag, diag, updiag]
            diags = [-1, 0, 1]
            TD_matrix = sparse.spdiags(mydata, diags, self.nlats, self.nlats,
                                       format='csc')

            rhs = (self.T * ((self.C / dt) - term1 - (self.B / 2)) +
                   T_p1 * ((term1 / 2) - term2) +
                   T_m1 * ((term1 / 2) + term2) - self.A + insolation)

            T_new = linalg.spsolve(TD_matrix, rhs)

            self.eps = np.sum(abs(self.T - T_new)**2)

            self.T = T_new
            self.T_mean = self.calc_mean_T()
            self.freeze_lat = self.calc_iceline()

            self.iter_T_means.append(self.T_mean)
            self.iter_freeze_lat.append(self.freeze_lat)

            self.niters = curr_iter
            if self.eps <= self.tol:
                break
        else:
            result += 'ERROR: Climate model did not converge on a solution.\n'

        result += 'Final mean temperature = {:2.2f}\n'.format(
            self.convert_degC_to_degF(self.T_mean))
        if self.freeze_lat is not None:
            result += 'Iceline latitude = {:2.1f}\n'.format(self.freeze_lat)

        result += 'Number of iterations {:d}'.format(self.niters)

        return result

    def calc_mean_T(self):
        return np.sum(np.cos(self.lats) * self.T) / np.sum(np.cos(self.lats))

    def calc_iceline(self):
        freezing = self.T <= -1.8
        if not np.any(freezing):
            # print('No freezing locations detected...')
            return None
        else:
            freeze_lat = np.min(np.abs(self.lats_in_deg[freezing]))
            return freeze_lat

    @staticmethod
    def convert_degC_to_degF(temp):
        return temp * 9 / 5 + 32

    def convert_1d_to_grid(self):

        lons, lats = np.meshgrid(self.lons_in_deg, self.lats_in_deg)
        values = np.ones_like(lons) * self.T[:, None]
        values = values.astype(np.float32)

        dataset = xr.DataArray(values, coords=[self.lats_in_deg,
                                               self.lons_in_deg],
                               dims=['lat', 'lon'])

        return dataset

class SimpleClimate(object):
    """
    Planetary climate class that only uses albedo and a column IR opacity to approximate surface temperature
    """
    
    def __init__(self,terra_sol_obj, plot_width=800,
                tau_star=0.84, f_cloud=0.7, A_cloud=0.4,
                f_land = 0.3, A_land=0.2):
        sigma = 5.670367e-8  # Stefan-Boltzmann Constant W.m^-2.K^-4
        self.S0 = terra_sol_obj.get_planet_energy_in()/4.
        self.alpha = f_cloud*A_cloud + (1-f_cloud)*f_land*A_land
        self.Ts = (((1-self.alpha)*self.S0*(1+.75*tau_star))/(4.*sigma))**.25t
    

if __name__ == '__main__':
    model = EnergyBalanceModel(Q=300)
    print(model.solve_climate())
    data = model.convert_1d_to_grid()
    x = 1
