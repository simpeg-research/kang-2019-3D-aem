from discretize import TensorMesh, TreeMesh
from SimPEG import utils
import numpy as np
import discretize
from pymatsolver import PardisoSolver

def create_local_mesh(
    src_location,
    rx_location,
    topo_location,
    h = [10., 10., 5.],
    x_core_lim = (-100., 100.),
    y_core_lim = (-20., 20.),
    padding_distance = [[4000., 4000.], [4000., 4000.], [4000., 4000.]]
    ):

    # TODO: All parameters used for generating this mesh should be input parameters
    # Currently fixed for a specific case

    xyz = np.vstack((rx_location, src_location))
    x = np.linspace(x_core_lim[0], x_core_lim[1]) + src_location[0]
    y = np.linspace(y_core_lim[0], y_core_lim[1]) + src_location[1]
    dem = utils.ndgrid(x, y, np.r_[topo_location[2]])

    mesh_local = discretize.utils.mesh_builder_xyz(
        dem,
        h,
        padding_distance=padding_distance,
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

    return mesh_local
    # actv_local = utils.surface2ind_topo(mesh_local, topo)
    # return mesh_local, actv_local