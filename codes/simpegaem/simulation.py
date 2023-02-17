import numpy as np
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from SimPEG.electromagnetics.time_domain import Simulation3DElectricField
from SimPEG.electromagnetics.time_domain.sources import StepOffWaveform, PiecewiseLinearWaveform
from scipy.constants import mu_0

def piecewise_ramp_fast(
    step_func, t_off, t_currents, currents, x, w,
    eps=1e-10
):
    """
    Computes response from piecewise linear current waveform
    with a single pulse. This basically evaluates the convolution
    between dI/dt and step-off response.

    step_func: function handle to evaluate step-off response
    t_off: time channels when the current is off
    t_shift: t_off + T/2
    currents: input source currents
    n: Gaussian quadrature order
    """
    n = x.size
    dt = np.diff(t_currents)
    dI = np.diff(currents)
    dIdt = dI/dt
    nt = t_currents.size
    pulse_time = t_currents.max()

    # Create a bunch of memory in C and use broadcasting
    t_lag = pulse_time - t_currents
    t_lag_expand = (np.repeat(t_lag[1:, np.newaxis], t_off.size, 1)).T
    t_lag_3D = np.repeat(t_lag_expand[:, :, np.newaxis], n, 2)
    t3D = t_lag_3D + t_off[:, np.newaxis, np.newaxis]
    # Gauss-Legendre part.
    # Expand time shifts and origin to 3D with G-L points
    inds = t3D[:,:,0] < 0.
    # Compute dt for both on-time and off-time
    # off-time f(t, t+t0)
    # on-time f(0, t+t0)
    dt_on_off = np.tile(dt, (t_off.size, 1))
    dt_on_off[inds] = (dt + t3D[:,:,0])[inds]
    t3D[inds,:] = 0.

    y = dt_on_off[:,:,np.newaxis] * (0.5 * (x + 1.0)) + t3D

    # Evaluate and weight G-L values with current waveform
    f = w * step_func(np.log10(y))
    s = f.sum(axis = 2) * 0.5 * dt_on_off

    response = np.sum(s * -dIdt, axis=1)

    return response

def piecewise_pulse_fast(
    step_func, t_off, t_currents, currents, n=20
):
    """
    Computes response from double pulses (negative then positive)
    T: Period (e.g. 25 Hz base frequency, 0.04 s period)
    """

    # Get gauss-legendre points and weights early since n never changes inside here
    x, w = roots_legendre(n)

    response = piecewise_ramp_fast(
            step_func, t_off, t_currents, currents, x, w
    )
    return response

class SimulationAEM(Simulation3DElectricField):
    """docstring for SimulationAEM"""

    def dpred(
        self,
        m,
        eps=1e-10,
    ):
        if self.verbose:
            print('{}\nSimulating SkyTEM data\n{}'.format('*'*50, '*'*50))

        self.model = m
        n_steps = self.time_steps.size
        factor = 3/2.
        n_src = self.survey.nSrc
        voltage_step_off = np.zeros(
            (self.time_steps.size, n_src), dtype=float, order='C'
        )

        sol_n0 = np.zeros((self.mesh.n_edges, n_src), dtype=float, order='F')
        sol_n1 = np.zeros((self.mesh.n_edges, n_src), dtype=float, order='F')
        sol_n2 = np.zeros((self.mesh.n_edges, n_src), dtype=float, order='F')
        s_e = np.zeros((self.mesh.n_edges, n_src), order='F')

        rx_locations = np.zeros((n_src, 3), order='F')

        # Generate initial conditions
        C = self.mesh.edgeCurl
        for i_src, src in enumerate(self.survey.source_list):
            b0 = src.bInitial(self)
            s_e[:, i_src] = C.T*self.MfMui*b0
            rx_locations[i_src, :] = src.receiver_list[0].locations

        # Assume only z-component
        # TODO: need to be generalized
        Fz = self.mesh.getInterpolationMat(rx_locations, locType='Fz')

        #  Time steps
        dt_0 = 0.
        for ii in range(n_steps):
            dt = self.time_steps[ii]
            # Factor for BDF2
            if abs(dt_0-dt) > eps:
                if ii != 0:
                    Ainv.clean()
                A = self.getAdiag(dt, factor=factor)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.solver(A, **self.solver_opts)
                if self.verbose:
                    print('Done')

            # TODO: Need to integrate in to RHS and getAsubdiag
            if ii == 0:
                rhs = factor/dt*s_e
            elif ii == 1:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol_n1+1/3.*sol_n0) + 1./3.*s_e
                )
            else:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol_n1+1/3.*sol_n0)
                )
            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(ii+1))

            sol_n2 = Ainv*rhs

            if self.verbose:
                print('    Done...')

            # Need data matrix
            if n_src > 1:
                voltage_step_off[ii, :] = (Fz*(-C*sol_n2)).diagonal()
            else:
                sol_n2 = sol_n2.reshape([-1, 1])
                voltage_step_off[ii] = (Fz*(-C*sol_n2))
            dt_0 = dt
            sol_n0 = sol_n1.copy()
            sol_n1 = sol_n2.copy()

        # clean factors and return
        Ainv.clean()
        data = []
        for i_src, src in enumerate(self.survey.source_list):
            step_func = interp1d(
                np.log10(self.time_mesh.gridCC[:]), voltage_step_off[:, i_src]
            )
            times = src.receiver_list[0].times
            if isinstance(src.waveform, StepOffWaveform):
                data_tmp = step_func(np.log10(times))
            elif isinstance(src.waveform, PiecewiseLinearWaveform):
                data_tmp = piecewise_pulse_fast(
                    step_func, times,
                    src.waveform.times,
                    src.waveform.currents
                )
            else:
                raise Exception(
                    "waveform type should be either StepOffWaveform or PiecewiseLinearWaveform"
                )
            data.append(data_tmp)
        return np.hstack(data)

    def getAdiag(self, dt, factor=1.):
        """
        Diagonal of the system matrix at a given time index
        """
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T * (MfMui * C) + factor/dt * MeSigma