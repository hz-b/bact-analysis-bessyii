from .fit import get_response
from .prepare_plotdata import extract_matrices
from ..model.analysis_model import FitReadyData
from ..model.analysis_util import flatten_for_fit
from .preprocess_data import load_and_rearrange_data
from .plot_matplotlib import plot_orms as mpl_plot_orms
from .plot_pyvista import plot_orms as pv_plot_orms
from matplotlib import pyplot as plt
import tqdm


def main(uid):
    preprocessed_measurement = load_and_rearrange_data(uid)
    fit_ready_data = FitReadyData(
        per_magnet=[
            flatten_for_fit(
                measurement.per_magnet, measurement.name, pos="pos_raw", rms="rms_raw"
            )
            for measurement in tqdm.tqdm(
                preprocessed_measurement.measurement,
                desc="prepare fit data: ",
                total=(len(preprocessed_measurement.measurement)),
            )
        ]
    )

    orms = extract_matrices(get_response(fit_ready_data))
    pv_plot_orms(orms)
    return

    try:
        mpl_plot_orms(orms)
    except Exception:
        raise
    else:
        plt.show()
