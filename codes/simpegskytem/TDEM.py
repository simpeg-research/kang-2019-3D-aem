import numpy as np
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import interp1d

from SimPEG import Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.TDEM import Problem3D_e
from simpegEM1D.Waveforms import piecewise_pulse_fast


class ProblemSkyTEM(Problem3D_e):
    """docstring for Problem3D"""

    def __init__(self, mesh, **kwargs):
        Problem3D_e.__init__(self, mesh, **kwargs)

    def simulate(
        self,
        m,
        time,
        time_dual_moment,
        time_input_currents,
        input_currents,
        time_input_currents_dual_moment,
        input_currents_dual_moment,
        eps=1e-10,
        base_frequency=30,
        base_frequency_dual_moment=210,

    ):
        if self.verbose:
            print('{}\nSimulating SkyTEM data\n{}'.format('*'*50, '*'*50))

        self.model = m
        n_steps = self.timeSteps.size
        factor = 3/2.
        nSrc = self.survey.nSrc
        data_stepoff = np.zeros(
            (self.timeSteps.size, nSrc), dtype=float, order='C'
        )

        sol_n0 = np.zeros((self.mesh.nE, nSrc), dtype=float, order='F')
        sol_n1 = np.zeros((self.mesh.nE, nSrc), dtype=float, order='F')
        sol_n2 = np.zeros((self.mesh.nE, nSrc), dtype=float, order='F')
        s_e = np.zeros((self.mesh.nE, nSrc), order='F')

        locs = np.zeros((nSrc, 3), order='F')

        # Generate initial conditions
        C = self.mesh.edgeCurl
        for i_src, src in enumerate(self.survey.srcList):
            b0 = src.bInitial(self)
            s_e[:, i_src] = C.T*self.MfMui*b0
            locs[i_src, :] = src.rxList[0].locs

        # Assume only z-component
        # TODO: need to be generalized
        Fz = self.mesh.getInterpolationMat(locs, locType='Fz')

        #  Time steps
        dt_0 = 0.
        for ii in range(n_steps):
            dt = self.timeSteps[ii]
            # Factor for BDF2
            if abs(dt_0-dt) > eps:
                if ii != 0:
                    Ainv.clean()
                A = self.getAdiag(dt, factor=factor)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose:
                    print('Done')

            # Need to integrate in to RHS and getAsubdiag
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
            if nSrc > 1:
                data_stepoff[ii, :] = (Fz*(-C*sol_n2)).diagonal()
            else:
                sol_n2 = sol_n2.reshape([-1, 1])
                data_stepoff[ii] = (Fz*(-C*sol_n2))
            dt_0 = dt
            sol_n0 = sol_n1.copy()
            sol_n1 = sol_n2.copy()

        # clean factors and return
        Ainv.clean()

        period = 1./base_frequency
        period_dual_moment = 1./base_frequency_dual_moment

        data = np.zeros((nSrc, time.size), float, 'C')
        data_dual_moment = np.zeros((nSrc, time_dual_moment.size), float, 'C')

        for ii in range(nSrc):
            step_func = interp1d(
                self.timeMesh.gridCC, data_stepoff[:, ii]
            )
            data_temp = piecewise_pulse_fast(
                                step_func, time,
                                time_input_currents, input_currents,
                                period, n_pulse=1
            )
            data_dual_moment_temp = piecewise_pulse_fast(
                                step_func, time_dual_moment,
                                time_input_currents_dual_moment, input_currents_dual_moment,
                                period_dual_moment, n_pulse=1
            )
            data[ii, :] = data_temp
            data_dual_moment[ii, :] = data_dual_moment_temp

        return np.r_[data.flatten(), data_dual_moment.flatten()]

    def getAdiag(self, dt, factor=1.):
        """
        Diagonal of the system matrix at a given time index
        """
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T * (MfMui * C) + factor/dt * MeSigma
