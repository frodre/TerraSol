# Climate model based off of Judy and Hansi's energy balance notebook
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import xarray as xr


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





if __name__ == '__main__':
    model = EnergyBalanceModel(Q=300)
    print(model.solve_climate())
    data = model.convert_1d_to_grid()
    x = 1