import numpy as np
from SimPEG import EM, Utils
from .TDEM import ProblemSkyTEM
from pymatsolver import Pardiso
import discretize
import dill

def create_local_mesh(
    src_location,
    rx_location,
    topo_location,
    topo,
    h = [10., 10., 5.],
    x_core_lim = (-100., 100.),
    y_core_lim = (-20., 20.),
    ):

    # TODO: All parameters used for generating this mesh should be input parameters
    # Currently fixed for a specific case

    xyz = np.vstack((rx_location, src_location))
    x = np.linspace(x_core_lim[0], x_core_lim[1]) + src_location[0]
    y = np.linspace(y_core_lim[0], y_core_lim[1]) + src_location[1]
    dem = Utils.ndgrid(x, y, np.r_[topo_location[2]])

    mesh_local = discretize.utils.mesh_builder_xyz(
        dem,
        h,
        padding_distance=[[2000., 2000.], [2000., 2000.], [2000., 2000.]],
        base_mesh=None,
        depth_core=None,
        expansion_factor=1.3,
        mesh_type='tree'
    )

    mesh_local = discretize.utils.refine_tree_xyz(
        mesh_local,
        dem,
        method='surface',
        octree_levels=[5, 10, 10],
        octree_levels_padding=None,
        finalize=False,
        min_level=0,
        max_distance=np.inf,
    )


    mesh_local = discretize.utils.refine_tree_xyz(
        mesh_local,
        xyz,
        method='radial',
        octree_levels=[2, 0, 0],
        octree_levels_padding=None,
        finalize=True,
        min_level=1,
        max_distance=np.inf,
    )

    actv_local = Utils.surface2ind_topo(mesh_local, topo)

    return mesh_local, actv_local

def run_simulation_skytem(args):
    """
    run_simulation_skytem
    --------------
    """
    from pyMKL import mkl_set_num_threads
    i_src, work_dir, n_thread = args
    mkl_set_num_threads(n_thread)
    inputs = dill.load(open(work_dir+"inputs_{}.pkl".format(i_src), 'rb'))
    mesh_local = inputs['mesh_local']
    actv_local = inputs['actv_local']
    sigma_interpolator = inputs['sigma_interpolator']
    srcloc = inputs['srcloc']
    rxloc = inputs['rxloc']
    time = inputs['time']
    time_input_currents = inputs['time_input_currents']
    input_currents = inputs['input_currents']
    base_frequency = inputs['base_frequency']
    time_dual_moment = inputs['time_dual_moment']
    time_input_currents_dual_moment = inputs['time_input_currents_dual_moment']
    input_currents_dual_moment = inputs['input_currents_dual_moment']
    base_frequency_dual_moment = inputs['base_frequency_dual_moment']

    values = sigma_interpolator(mesh_local.gridCC[actv_local,:])
    sigma_local = np.ones(mesh_local.nC) * 1e-8
    sigma_local[actv_local] = values

    rx = EM.TDEM.Rx.Point_dbdt(rxloc, np.logspace(np.log10(1e-6), np.log10(1e-2), 31), 'z')
    src = EM.TDEM.Src.MagDipole([rx], waveform=EM.TDEM.Src.StepOffWaveform(), loc=srcloc)
    survey = EM.TDEM.Survey([src])
    prb = ProblemSkyTEM(mesh_local, sigma=sigma_local, verbose=False)
    dts = np.diff(np.logspace(-6, -1, 50))
    prb.timeSteps = [
        (3e-7, 6),(1e-6, 5),(2e-6, 5),(5e-6, 5),
        (1e-5, 5),(2e-5, 5),(5e-5, 5),(1e-4, 5),
        (2e-4, 5),(5e-4, 5),(1e-3, 15)
    ]
    prb.Solver = Pardiso
    prb.pair(survey)
    data = prb.simulate(
            sigma_local,
            time,
            time_dual_moment,
            time_input_currents,
            input_currents,
            time_input_currents_dual_moment,
            input_currents_dual_moment,
            base_frequency=base_frequency,
            base_frequency_dual_moment=base_frequency_dual_moment
    )
    return data
