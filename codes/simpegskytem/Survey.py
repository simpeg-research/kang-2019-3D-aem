import numpy as np
from SimPEG import Utils, Survey
import properties


class GlobalAEMSurvey(Survey.BaseSurvey, properties.HasProperties):

    # This assumes a multiple sounding locations
    rx_locations = properties.Array(
        "Receiver locations ", dtype=float, shape=('*', 3)
    )
    src_locations = properties.Array(
        "Source locations ", dtype=float, shape=('*', 3)
    )
    topo = properties.Array(
        "Topography", dtype=float, shape=('*', 3)
    )

    half_switch = properties.Bool("Switch for half-space", default=False)

    _pred = None



    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.prob.fields is called.
        """
        if f is None:
            f = self.prob.fields(m)

        return self._pred

    @property
    def n_sounding(self):
        """
            # of Receiver locations
        """
        return self.rx_locations.shape[0]

    def read_xyz_data(self, fname):
        """
        Read csv file format
        This is a place holder at this point
        """
        pass

    @property
    def nD(self):
        # Need to generalize this for the dual moment data
        if getattr(self, '_nD', None) is None:
            self._nD = self.nD_vec.sum()
        return self._nD

class GlobalAEMSurveyTD(GlobalAEMSurvey):

    # --------------- Essential inputs ---------------- #
    src_type = None

    rx_type = None

    field_type = None

    time = []

    wave_type = None

    moment_type = None

    moment = None

    time_input_currents = []

    input_currents = []

    # --------------- Selective inputs ---------------- #
    n_pulse = properties.Array(
        "The number of pulses",
        default=None
    )

    base_frequency = properties.Array(
        "Base frequency (Hz)",
        dtype=float, default=None
    )

    offset = properties.Array(
        "Src-Rx offsets", dtype=float, default=None,
        shape=('*', '*')
    )

    I = properties.Array(
        "Src loop current", dtype=float, default=None
    )

    radius = properties.Array(
        "Src loop radius", dtype=float, default=None
    )

    use_lowpass_filter = properties.Array(
        "Switch for low pass filter",
        dtype=bool, default=None
    )

    high_cut_frequency = properties.Array(
        "High cut frequency for low pass filter (Hz)",
        dtype=float, default=None
    )

    # ------------- For dual moment ------------- #

    time_dual_moment = []

    time_input_currents_dual_moment = []

    input_currents_dual_moment = []

    base_frequency_dual_moment = properties.Array(
        "Base frequency for the dual moment (Hz)",
        dtype=float, default=None
    )

    def __init__(self, **kwargs):
        GlobalAEMSurvey.__init__(self, **kwargs)
        self.set_parameters()

    def set_parameters(self):
        # TODO: need to put some validation process
        # e.g. for VMD `offset` must be required
        # e.g. for CircularLoop `a` must be required

        print(">> Set parameters")
        if self.n_pulse is None:
            self.n_pulse = np.ones(self.n_sounding, dtype=int) * 1

        if self.base_frequency is None:
            self.base_frequency = np.ones(
                (self.n_sounding), dtype=float
            ) * 30

        if self.offset is None:
            self.offset = np.empty((self.n_sounding, 1), dtype=float)

        if self.moment is None:
            self.moment = np.ones(self.n_sounding, dtype=float)

        if self.radius is None:
            self.radius = np.empty(self.n_sounding, dtype=float)

        if self.use_lowpass_filter is None:
            self.use_lowpass_filter = np.zeros(self.n_sounding, dtype=bool)

        if self.high_cut_frequency is None:
            self.high_cut_frequency = np.empty(self.n_sounding, dtype=float)

        if self.moment_type is None:
            self.moment_type = np.array(["single"], dtype=str).repeat(
                self.n_sounding, axis=0
            )

        # List
        if not self.time_input_currents:
            self.time_input_currents = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.input_currents:
            self.input_currents = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]

        # List
        if not self.time_dual_moment:
            self.time_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.time_input_currents_dual_moment:
            self.time_input_currents_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.input_currents_dual_moment:
            self.input_currents_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]

        if self.base_frequency_dual_moment is None:
            self.base_frequency_dual_moment = np.empty(
                (self.n_sounding), dtype=float
            )

    @property
    def nD_vec(self):
        if getattr(self, '_nD_vec', None) is None:
            self._nD_vec = []

            for ii, moment_type in enumerate(self.moment_type):
                if moment_type == 'single':
                    self._nD_vec.append(self.time[ii].size)
                elif moment_type == 'dual':
                    self._nD_vec.append(
                        self.time[ii].size+self.time_dual_moment[ii].size
                    )
                else:
                    raise Exception("moment_type must be either signle or dual")
            self._nD_vec = np.array(self._nD_vec)
        return self._nD_vec

    @property
    def data_index(self):
        # Need to generalize this for the dual moment data
        if getattr(self, '_data_index', None) is None:
            self._data_index = [
                    np.arange(self.nD_vec[i_sounding])+np.sum(self.nD_vec[:i_sounding]) for i_sounding in range(self.n_sounding)
            ]
        return self._data_index

    @property
    def nD(self):
        # Need to generalize this for the dual moment data
        if getattr(self, '_nD', None) is None:
            self._nD = self.nD_vec.sum()
        return self._nD
