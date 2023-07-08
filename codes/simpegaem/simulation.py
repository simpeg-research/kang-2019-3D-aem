import numpy as np
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from SimPEG.electromagnetics.time_domain import Simulation3DElectricField
from SimPEG.electromagnetics.time_domain.sources import StepOffWaveform
from scipy.constants import mu_0
from SimPEG import utils
from discretize import TensorMesh


# No need to have these functions
# def piecewise_ramp_fast(
#     step_func, t_off, t_currents, currents, x, w,
#     eps=1e-10
# ):
#     """
#     Computes response from piecewise linear current waveform
#     with a single pulse. This basically evaluates the convolution
#     between dI/dt and step-off response.

#     step_func: function handle to evaluate step-off response
#     t_off: time channels when the current is off
#     t_shift: t_off + T/2
#     currents: input source currents
#     n: Gaussian quadrature order
#     """
#     n = x.size
#     dt = np.diff(t_currents)
#     dI = np.diff(currents)
#     dIdt = dI/dt
#     nt = t_currents.size
#     pulse_time = t_currents.max()

#     # Create a bunch of memory in C and use broadcasting
#     t_lag = pulse_time - t_currents
#     t_lag_expand = (np.repeat(t_lag[1:, np.newaxis], t_off.size, 1)).T
#     t_lag_3D = np.repeat(t_lag_expand[:, :, np.newaxis], n, 2)
#     t3D = t_lag_3D + t_off[:, np.newaxis, np.newaxis]
#     # Gauss-Legendre part.
#     # Expand time shifts and origin to 3D with G-L points
#     inds = t3D[:,:,0] < 0.
#     # Compute dt for both on-time and off-time
#     # off-time f(t, t+t0)
#     # on-time f(0, t+t0)
#     dt_on_off = np.tile(dt, (t_off.size, 1))
#     dt_on_off[inds] = (dt + t3D[:,:,0])[inds]
#     t3D[inds,:] = 0.

#     y = dt_on_off[:,:,np.newaxis] * (0.5 * (x + 1.0)) + t3D

#     # Evaluate and weight G-L values with current waveform
#     f = w * step_func(np.log10(y))
#     s = f.sum(axis = 2) * 0.5 * dt_on_off

#     response = np.sum(s * -dIdt, axis=1)

#     return response

# def piecewise_pulse_fast(
#     step_func, t_off, t_currents, currents, n=20
# ):
#     """
#     Computes response from double pulses (negative then positive)
#     T: Period (e.g. 25 Hz base frequency, 0.04 s period)
#     """

#     # Get gauss-legendre points and weights early since n never changes inside here
#     x, w = roots_legendre(n)

#     response = piecewise_ramp_fast(
#             step_func, t_off, t_currents, currents, x, w
#     )
#     return response


class SimulationAEM(Simulation3DElectricField):

    _convolution_matricies_set = False


    def _compute_evaluation_matricies(self):
        """
            Store spatial and time evaluation matrices. 
        """
        if self._convolution_matricies_set:
            return 

        survey = self.survey

        Pts = []
        Pss = []
        times_step = self.time_mesh.cell_centers
        C = self.mesh.edge_curl

        # TODO: Generalize this to handle multiple receivers.
        for src in survey.source_list:
            # Assume there is a single receiver
            rx = src.receiver_list[0]
            wave = src.waveform
            if isinstance(wave, StepOffWaveform):
                log10_time = np.log10(self.time_mesh.cell_centers)
                ht = np.diff(log10_time)
                log10_time_mesh = TensorMesh([ht], x0=[log10_time[0]])
                A = log10_time_mesh.get_interpolation_matrix(np.log10(rx.times), location_type='N')             
            else:
                t_min = np.infty
                t_max = -np.infty
                x, w = roots_legendre(251)
                times = rx.times - wave.time_nodes[:, None]
                times[times < 0.0] = 0.0
                quad_points = (times[:-1] - times[1:])[..., None] * (
                    x + 1
                ) + times[1:, :, None]
                t_min = min(quad_points[quad_points > 0].min(), t_min)
                t_max = max(quad_points[quad_points > 0].max(), t_max)
                
                n_t = len(times_step)
                splines = []
                for i in range(n_t):
                    e = np.zeros(n_t)
                    e[i] = 1.0
                    sp = iuSpline(np.log(times_step), e, k=5)
                    splines.append(sp)
                # As will go from frequency to time domain

                def func(t, i):
                    out = np.zeros_like(t)
                    t = t.copy()
                    t[
                        (t > 0.0) & (t <= times_step.min())
                    ] = times_step.min()  # constant at very low ts
                    out[t > 0.0] = splines[i](np.log(t[t > 0.0])) 
                    # / t[t > 0.0]
                    return out

                # Then calculate the values at each time
                A = np.zeros((len(rx.times), n_t))
                # loop over pairs of nodes and use gaussian quadrature to integrate

                time_nodes = wave.time_nodes
                n_interval = len(time_nodes) - 1
                quad_times = []
                for i in range(n_interval):
                    b = rx.times - time_nodes[i]
                    b = np.maximum(b, 0.0)
                    a = rx.times - time_nodes[i + 1]
                    a = np.maximum(a, 0.0)
                    quad_times = (b - a)[:, None] * (x + 1) / 2.0 + a[:, None]
                    quad_scale = (b - a) / 2
                    wave_eval = wave.eval_deriv(rx.times[:, None] - quad_times)
                    for i in range(n_t):
                        A[:, i] -= np.sum(
                            quad_scale[:, None]
                            * w
                            * wave_eval
                            * func(quad_times, i),
                            axis=-1,
                        )
            Pts.append(A)
            # Assume only a z-component
            Pss.append(-self.mesh.get_interpolation_matrix(rx.locations, location_type='Fz') @ C)
        self._convolution_matricies_set = True 
        self._Pts = Pts
        self._Pss = Pss

    # TODO: Need to think about how to store fields?
    # Assume a single source 
    def fields(self, m):

        if self.verbose:
            print('{}\nSimulating time-domain Airborne EM data\n{}'.format('*'*50, '*'*50))

        self.model = m
        self._compute_evaluation_matricies()
        n_steps = self.time_steps.size
        factor = 3/2.
        eps = 1e-12
        n_src = self.survey.nSrc

        sol = np.zeros((self.mesh.n_edges, n_src, n_steps), dtype=float, order='F')
        s_e = np.zeros((self.mesh.n_edges, n_src), order='F')

        # Generate initial conditions
        C = self.mesh.edge_curl
        for i_src, src in enumerate(self.survey.source_list):
            b0 = src.bInitial(self)
            s_e[:, i_src] = C.T*self.MfMui*b0

        #  Time steps
        dt_0 = 0.
        for tInd in range(n_steps):
            dt = self.time_steps[tInd]
            # Factor for BDF2
            if abs(dt_0-dt) > eps:
                if tInd != 0:
                    Ainv.clean()
                A = self.getAdiag(tInd)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.solver(A, **self.solver_opts)
                if self.verbose:
                    print('Done')

            # TODO: Need to integrate in to RHS and getAsubdiag
            if tInd == 0:
                rhs = factor/dt*s_e
            elif tInd == 1:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol[:, :, tInd-1]+1/3.*sol[:, :, tInd-2]) + 1./3.*s_e
                )
            else:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol[:, :, tInd-1]+1/3.*sol[:, :, tInd-2])
                )
            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(tInd+1))
            if n_src == 1:
                sol[:, :, tInd] = (Ainv*rhs).reshape([-1,1])
            else:
                sol[:, :, tInd] = Ainv*rhs

            if self.verbose:
                print('    Done...')

        return sol

    def dpred(self, m, f=None):
        
        self.model = m
        if f is None:
            f = self.fields(m)
    
        data = []
        for isrc, src in enumerate(self.survey.source_list):
            step_response = (self._Pss[isrc] @ f[:,isrc,:]).flatten()
            data.append(self._Pts[isrc] @ step_response)

        return np.hstack(data)
    
    # def Jvec(m, v=None):
    #     J_sigma  = self.getJ_sigma(m)
    #     return  self_J_sigma @ (self.sigmaMap.Deriv @ v)
    
    # def Jtvec:
    #     J_sigma = self.getJ_sigma(m)
    #     return  self.sigmaMap.Deriv.T @ (J_sigma.T @ v)
    
    def getJ_sigma(self, m, f=None):
        
        self.model = m

        if f is None:
            f = self.fields(m)

        eps = 1e-10
        n_steps = self.time_steps.size
        nE = self.mesh.n_edges
        nD = self.survey.nD
        nT = len(self.time_steps)
        nM = self.mesh.n_cells
        dt_0 = 0.
        survey = self.survey

        J_matrixT = np.zeros((nM, nD), dtype=float, order='F')

        for i_src, src in enumerate(survey.source_list):
            i_start = survey.vnD[:i_src].sum()
            i_end = i_start + survey.vnD[i_src]

            yn = np.zeros((nE, src.nD), dtype=float, order='F')
            yn_1 = np.zeros((nE, src.nD), dtype=float, order='F')
            yn_2 = np.zeros((nE, src.nD), dtype=float, order='F')
         
            for tInd in reversed(range(n_steps)):
                # print (tInd, tInd-1, tInd-2)
                dt = self.time_steps[tInd]
                A = self.getAdiag(tInd)
                if abs(dt_0-dt) > eps:
                    # print (tInd)
                    Ainv = self.solver(A)

                pn = (self._Pss[i_src].T).toarray() * (self._Pts[i_src].T[tInd,:].reshape([1,-1]))
                if tInd==n_steps-1:
                    yn = Ainv * pn
                elif tInd==n_steps-2: 
                    BB = self.getAsubdiag(tInd+1)
                    yn = Ainv * (pn-BB@yn_1)   
                else:
                    BB = self.getAsubdiag(tInd+1)
                    CC = self.getAsubsubdiag(tInd+2)
                    yn = Ainv * (pn-BB@yn_1 -CC@yn_2)   
                dAT_dm = self.getAdiagDeriv(tInd, f[:,i_src,tInd], v=None, adjoint=True) @ yn
                if tInd == 0:
                    dAsubdiagT_dm = utils.Zero()
                    dAsubsubdiagT_dm = utils.Zero()
                elif tInd == 1:
                    dAsubdiagT_dm = self.getAsubdiagDeriv(tInd, f[:,i_src,tInd-1], v=None, adjoint=True) @ yn
                    dAsubsubdiagT_dm = utils.Zero()
                else:
                    dAsubdiagT_dm = self.getAsubdiagDeriv(tInd, f[:,i_src,tInd-1], v=None, adjoint=True) @ yn
                    dAsubsubdiagT_dm = self.getAsubsubdiagDeriv(tInd, f[:,i_src,tInd-2], v=None, adjoint=True) @ yn
                J_matrixT[:,i_start:i_end] = J_matrixT[:,i_start:i_end] + - dAT_dm - dAsubdiagT_dm - dAsubsubdiagT_dm

                yn_2 = yn_1
                yn_1 = yn
                dt_0 = dt  

        return J_matrixT.T      


    def dpred_no_store(
        self,
        m,
        eps=1e-10,
    ):
        if self.verbose:
            print('{}\nSimulating time-domain Airborne EM data\n{}'.format('*'*50, '*'*50))

        self.model = m
        self._compute_evaluation_matricies()

        factor = 3/2.
        n_steps = self.time_steps.size
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
        C = self.mesh.edge_curl
        for i_src, src in enumerate(self.survey.source_list):
            b0 = src.bInitial(self)
            s_e[:, i_src] = C.T*self.MfMui*b0
            rx_locations[i_src, :] = src.receiver_list[0].locations

        # Assume only z-component
        # TODO: need to be generalized
        Fz = self.mesh.get_interpolation_matrix(rx_locations, location_type='Fz')

        #  Time steps
        dt_0 = 0.
        for tInd in range(n_steps):
            dt = self.time_steps[tInd]
            # Factor for BDF2
            if abs(dt_0-dt) > eps:
                if tInd != 0:
                    Ainv.clean()
                A = self.getAdiag(tInd)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.solver(A, **self.solver_opts)
                if self.verbose:
                    print('Done')

            # TODO: Need to integrate in to RHS and getAsubdiag
            if tInd == 0:
                rhs = factor/dt*s_e
            elif tInd == 1:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol_n1+1/3.*sol_n0) + 1./3.*s_e
                )
            else:
                rhs = -factor/dt*(
                    self.MeSigma*(-4/3.*sol_n1+1/3.*sol_n0)
                )
            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(tInd+1))

            sol_n2 = Ainv*rhs

            if self.verbose:
                print('    Done...')

            # Need data matrix
            if n_src > 1:
                voltage_step_off[tInd, :] = (Fz*(-C*sol_n2)).diagonal()
            else:
                sol_n2 = sol_n2.reshape([-1, 1])
                voltage_step_off[tInd] = (Fz*(-C*sol_n2))
            dt_0 = dt
            sol_n0 = sol_n1.copy()
            sol_n1 = sol_n2.copy()

        # clean factors and return
        Ainv.clean()
        data = []
        for i_src, src in enumerate(self.survey.source_list):
            data_tmp = self._Pts[i_src] @ voltage_step_off[:, i_src]
            data.append(data_tmp)
        return np.hstack(data)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfMui = self.MfMui
        MeSigma = self.MeSigma
        return C.T.tocsr() * (MfMui * C) + (3./2.) / dt * MeSigma

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Deriv of ADiag with respect to electrical conductivity
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        # MeSigmaDeriv = self.MeSigmaDeriv(u)

        return (3./2.) / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        if tInd == 0:
            factor = -3./2.
        else:
            factor = -2.
        return factor / dt * self.MeSigma

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of the matrix below the diagonal with respect to electrical
        conductivity
        """
        dt = self.time_steps[tInd]
        if tInd == 0:
            factor = -3./2.
        else:
            factor = -2.
        if adjoint:
            return factor / dt * self.MeSigmaDeriv(u, v, adjoint)

        return factor / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getAsubsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]

        return 0.5 / dt * self.MeSigma

    def getAsubsubdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of the matrix below the diagonal with respect to electrical
        conductivity
        """
        dt = self.time_steps[tInd]

        if adjoint:
            return 0.5 / dt * self.MeSigmaDeriv(u, v, adjoint)

        return 0.5 / dt * self.MeSigmaDeriv(u, v, adjoint)        