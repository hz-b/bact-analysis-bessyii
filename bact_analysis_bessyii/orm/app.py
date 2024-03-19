from .fit import get_response_one_magnet
from .model import FitResultAllMagnets
from .prepare_plotdata import extract_response_matrices_per_steerers
from ..model.analysis_model import FitReadyData
from ..model.analysis_util import flatten_for_fit
from ..tools.preprocess_data import load_and_rearrange_data
from .plot_matplotlib import plot_orms as mpl_plot_orms
from .plot_pyvista import plot_orms as pv_plot_orms
from matplotlib import pyplot as plt
import tqdm


def main(uid):
    preprocessed_measurement = load_and_rearrange_data(
        uid,
        prefix="bessyii-orbit-response-measured",
        pv_for_applied_current="mux_sel_p_setpoint",
        pv_for_selected_magnet="mux_sel_selected",
        read_from_file=True,
    )
    fit_ready_data = FitReadyData(
        per_magnet=[
            flatten_for_fit(
                measurement.per_magnet, measurement.name, pos="pos_raw", rms="rms_raw"
            )
            for measurement in tqdm.tqdm(
                preprocessed_measurement.measurement,
                desc="prepare / rearrange  data for fit: ",
                total=(len(preprocessed_measurement.measurement)),
            )
        ]
    )

    fit_results = FitResultAllMagnets(
        data=[
            get_response_one_magnet(item)
            for item in tqdm.tqdm(
                fit_ready_data.per_magnet,
                desc="steerer on bpms fit:",
                total=len(fit_ready_data.per_magnet),
            )
        ]
    )

    orms = extract_response_matrices_per_steerers(fit_results)
    pv_plot_orms(orms, scale_bpm_readings=5e-4)

    try:
        mpl_plot_orms(orms)
    except Exception:
        raise
    else:
        plt.show()
