import numpy as np
import discretize
from SimPEG import maps, utils, props
from SimPEG.simulation import BaseSimulation
from SimPEG.base import BaseElectricalPDESimulation
import SimPEG.electromagnetics.time_domain as tdem
import properties
from pymatsolver import PardisoSolver
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from multiprocessing import Pool

from .simulation import SimulationAEM
from .utility import create_local_mesh


def run_simulation_time_domain(args):
    from pyMKL import mkl_set_num_threads
    mkl_set_num_threads(1)
    source, sigma_local, mesh_local, time_steps = args
    survey = tdem.Survey([source])
    simulation_3d = SimulationAEM(
        mesh=mesh_local,
        survey=survey,
        sigmaMap=maps.IdentityMap(mesh_local),
        solver=PardisoSolver,
        time_steps=time_steps
    )
    dpred = simulation_3d.dpred(sigma_local)
    return dpred

class GlobalSimulationAEM(BaseSimulation):
    """
    Base class for the stitched 1D simulation. This simulation models the EM
    response for a set of 1D EM soundings.
    """

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity at infinite frequency (S/m)")

    def __init__(
        self, mesh=None, survey=None, sigma=None, sigmaMap=None,
        topo=None, time_steps=None, parallel=None, n_cpu=None, **kwargs
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigma = sigma
        self.sigmaMap = sigmaMap
        self.topo = topo
        self.time_steps = time_steps
        self.parallel = parallel
        self.n_cpu = n_cpu

        if self.parallel:
            print(">> Use multiprocessing for parallelization")
            if self.n_cpu is None:
                self.n_cpu = multiprocessing.cpu_count()
            print((">> n_cpu: %i") % (self.n_cpu))
        else:
            print(">> Serial version is used")

    def input_args(self, i_src):
        args = (
            self.survey.source_list[i_src],
            self.sigma_locals[i_src],
            self.mesh_locals[i_src],
            self.time_steps
        )
        return args

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        if self.parallel:
            if self.verbose:
                print ('parallel')

            pool = Pool(self.n_cpu)
            result = pool.map(
                run_simulation_time_domain,
                [
                    self.input_args(i_src) for i_src in range(self.n_src)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation_time_domain(self.input_args(i_src)) for i_src in range(self.n_src)
            ]
        return np.hstack(result)

    @property
    def n_src(self):
        return self.survey.nSrc

    @property
    def f_topo(self):
        if getattr(self, '_f_topo', None) is None:
            try:
                self._f_topo = LinearNDInterpolator(self.topo[:,:2], self.topo[:,2])
            except:
                self._f_topo = NearestNDInterpolator(self.topo[:,:2], self.topo[:,2])
        return self._f_topo

    @property
    def sigma_locals(self):
        if getattr(self, '_sigma_locals', None) is None:
            # Ordering: first z then x
            self._sigma_locals = self._get_local_sigmas()
        return self._sigma_locals

    @property
    def P_global_to_locals(self):
        if getattr(self, '_P_global_to_locals', None) is None:
            # Ordering: first z then x
            self._P_global_to_locals = self._get_P_global_to_locals()
        return self._P_global_to_locals

    @property
    def mesh_locals(self):
        if getattr(self, '_mesh_locals', None) is None:
            # Ordering: first z then x
            self._mesh_locals = self._get_local_meshes()
        return self._mesh_locals

    # Below operations an also be parallelized if needed
    # But, this is a single time operation
    def _get_local_meshes(self):
        mesh_locals = []
        for i_src, src in enumerate(self.survey.source_list):
            source_location = src.location
            receiver_location = src.receiver_list[0].locations.flatten()
            topo_location = np.r_[source_location[:2], self.f_topo(source_location[:2])]
            mesh_local = create_local_mesh(
                source_location,
                receiver_location,
                topo_location
            )
            mesh_locals.append(mesh_local)
        return mesh_locals

    def _get_P_global_to_locals(self):
        P_global_to_locals = []
        for i_src in range(self.n_src):
            mesh_local = self.mesh_locals[i_src]
            P_global_to_local = discretize.utils.volume_average(self.mesh, mesh_local)
            P_global_to_locals.append(P_global_to_local)
        return P_global_to_locals

    def _get_local_sigmas(self):
        sigma_locals = []
        for i_src in range(self.n_src):
            sigma_local = self.P_global_to_locals[i_src] @ self.sigma
            sigma_locals.append(sigma_local)
        return sigma_locals

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_sigma_locals']
        return toDelete