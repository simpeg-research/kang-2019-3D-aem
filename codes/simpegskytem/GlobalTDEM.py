import numpy as np
from scipy.interpolate import RegularGridInterpolator
import dill
from SimPEG import Problem, Props, Utils, Maps, Survey
from .EMSimulation import create_local_mesh, run_simulation_skytem
import properties
import warnings
import os
import multiprocess
from multiprocess import Pool
import warnings
import dask.bag as db

warnings.filterwarnings("ignore")

class GlobalAEM(Problem.BaseProblem):
    """docstring for GlobalAEM"""

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    surveyPair = Survey.BaseSurvey  #: The survey to pair with.
    dataPair = Survey.Data  #: The data to pair with.
    mapPair = Maps.IdentityMap  #: Type of mapping to pair with

    actv = None
    work_dir = None
    sigma_fill_value = 1./20.
    n_cpu = None
    verbose = False
    n_thread = None
    parallel_option = 'multiprocess'


    def check_regular_mesh(self):

        if self.mesh._meshType != 'TENSOR':
            raise Exception("Mesh type must be TENSOR")

        h_uniq = np.unique(np.r_[self.mesh.hx, self.mesh.hy, self.mesh.hz])

        if h_uniq.size > 3:
            raise Exception("Dimensions of entire cells should be same (regular grid)")

    # ------------- For survey ------------- #
    @property
    def n_sounding(self):
        return self.survey.n_sounding

    @property
    def rx_locations(self):
        return self.survey.rx_locations

    @property
    def src_locations(self):
        return self.survey.src_locations

    @property
    def data_index(self):
        return self.survey.data_index

    @property
    def topo(self):
        return self.survey.topo

    @property
    def radius(self):
        return self.survey.radius

    @property
    def moment(self):
        return self.survey.moment

    @property
    def field_type(self):
        return self.survey.field_type

    @property
    def rx_type(self):
        return self.survey.rx_type

    @property
    def src_type(self):
        return self.survey.src_type

    def simulate(sefl, m):
        # return np.hstack(result)
        pass

    def clean_work_dir(self):
        import shutil
        try:
            shutil.rmtree(self.work_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))        
                # os.remove(self.work_dir + "*.pkl")
        # os.system( "rm -rf " + self.work_dir + "*.pkl")


class GlobalSkyTEM(GlobalAEM):

    def __init__(self, mesh, **kwargs):
        GlobalAEM.__init__(self, mesh, **kwargs)
        self.check_regular_mesh()
        if self.work_dir is None:
            self.work_dir = "./tmp/"
            os.mkdir(self.work_dir)

        if self.n_cpu is None:
            self.n_cpu =  multiprocess.cpu_count()

        self.n_thread = int(1)

    # ------------- For survey ------------- #
    @property
    def wave_type(self):
        return self.survey.wave_type

    @property
    def input_currents(self):
        return self.survey.input_currents

    @property
    def time_input_currents(self):
        return self.survey.time_input_currents

    @property
    def n_pulse(self):
        return self.survey.n_pulse

    @property
    def base_frequency(self):
        return self.survey.base_frequency

    @property
    def time(self):
        return self.survey.time

    @property
    def use_lowpass_filter(self):
        return self.survey.use_lowpass_filter

    @property
    def high_cut_frequency(self):
        return self.survey.high_cut_frequency

    @property
    def moment_type(self):
        return self.survey.moment_type

    @property
    def time_dual_moment(self):
        return self.survey.time_dual_moment

    @property
    def time_input_currents_dual_moment(self):
        return self.survey.time_input_currents_dual_moment

    @property
    def input_currents_dual_moment(self):
        return self.survey.input_currents_dual_moment

    @property
    def base_frequency_dual_moment(self):
        return self.survey.base_frequency_dual_moment

    @property
    def sigma_interpolator(self):
        if getattr(self, '_sigma_interpolator', None) is None:
            self._sigma_interpolator = RegularGridInterpolator(
                (self.mesh.vectorCCx, self.mesh.vectorCCy, self.mesh.vectorCCz),
                self.sigma_padded.reshape(self.mesh.vnC, order='F'),
                bounds_error=False, fill_value=self.sigma_fill_value
            )
        return self._sigma_interpolator

    @property
    def sigma_padded(self):
        if getattr(self, '_sigma_padded', None) is None:
            if self.actv.sum() != self.mesh.nC:
                ACTV = self.actv.reshape(
                    (np.prod(self.mesh.vnC[:2]), self.mesh.vnC[2]),
                    order='F'
                )
                index_at_topo = (
                    np.arange(np.prod(self.mesh.vnC[:2])) +
                    np.prod(self.mesh.vnC[:2]) * (ACTV.sum(axis=1)-1)
                )
                index_at_topo_up = (
                    np.arange(np.prod(self.mesh.vnC[:2])) +
                    np.prod(self.mesh.vnC[:2]) * (ACTV.sum(axis=1))
                )
                self._sigma_padded = self.sigma.copy()
                self._sigma_padded[index_at_topo_up] = self._sigma_padded[index_at_topo]
            else:
               self._sigma_padded  = self.sigma.copy()
        return self._sigma_padded

    def write_inputs_on_disk(self, i_src):

        # TODO: There can be two different definition of topography
        # - One at sounding locations
        # - The other for a general DEM
        # We can use DEM for better discretization of topography

        mesh_local, actv_local = create_local_mesh(
            self.src_locations[i_src,:],
            self.rx_locations[i_src,:],
            self.topo[i_src,:],
            self.topo
        )
        storage = {
            'srcloc': self.src_locations[i_src,:],
            'rxloc': self.rx_locations[i_src,:],
            'mesh_local': mesh_local,
            'actv_local': actv_local,
            'sigma_interpolator': self.sigma_interpolator,
            'time': self.time[i_src],
            'time_input_currents': self.time_input_currents[i_src],
            'input_currents': self.input_currents[i_src],
            'base_frequency': self.base_frequency[i_src],
            'time_dual_moment': self.time_dual_moment[i_src],
            'time_input_currents_dual_moment': self.time_input_currents_dual_moment[i_src],
            'input_currents_dual_moment': self.input_currents_dual_moment[i_src],
            'base_frequency_dual_moment': self.base_frequency_dual_moment[i_src],
        }
        dill.dump(
            storage, open(self.work_dir+"inputs_{}.pkl".format(i_src), 'wb')
        )

    def run_simulation(self, i_src):
        if self.verbose:
            print(">> Time-domain")
        return run_simulation_skytem(i_src)

    def write_inputs_on_disk_pool(self):

        n_src = self.n_sounding

        if self.parallel_option == 'multiprocess':
            pool = Pool(self.n_cpu)
            out = pool.map(
                self.write_inputs_on_disk,
                [i_src for i_src in range(n_src)]
            )
            pool.close()
            pool.join()

        elif self.parallel_option == 'dask':
            b = db.from_sequence(
                range(self.n_sounding), npartitions=np.ceil(self.n_sounding / self.n_cpu)
            )
            b = b.map(self.write_inputs_on_disk)
            out = b.compute()

    def forward(self, m):

        self.model = m

        if self.verbose:
            print(">> Compute response")
        if self.parallel_option == 'multiprocess':
            pool = Pool(self.n_cpu)
            # This assumes the same # of layer for each of soundings
            results = pool.map(
                run_simulation_skytem,
                [(i_src, self.work_dir, self.n_thread) for i_src in range(self.n_sounding)]
            )
            pool.close()
            pool.join()

        elif self.parallel_option == 'dask':
            b = db.from_sequence(
                [(i_src, self.work_dir, self.n_thread) for i_src in range(self.n_sounding)],
                npartitions=np.ceil(self.n_sounding / self.n_cpu)
            )
            b = b.map(run_simulation_skytem)
            results = b.compute()
        return np.hstack(results)
