from .fit import get_response_one_magnet
from .model import FitResultAllMagnets
from .prepare_plotdata import extract_response_matrices_per_steerers
from ..model.analysis_model import FitReadyData, MeasurementData, MeasurementPerMagnet
from ..model.analysis_util import flatten_for_fit
from ..steerer_response.bessyii_info_repos import DeviceLocationServiceBESSYII, BessyIIELementFamilies
from ..steerer_response.bessyii_lattice import twiss_from_at
from ..tools.preprocess_data import load_and_rearrange_data
from .plot_matplotlib import plot_orms as mpl_plot_orms
from .plot_pyvista import plot_orms as pv_plot_orms
from ..steerer_response.bessyii_info_repos import BESSYIIFamilyNames
from matplotlib import pyplot as plt
import tqdm


def main(uid):
    model = twiss_from_at()
    element_families = BessyIIELementFamilies(
        [datum.name for datum in model.survey.positions]
    )

    preprocessed_measurement = load_and_rearrange_data(
        uid,
        prefix="bessyii-orbit-response-measured",
        pv_for_applied_current="mux_sel_p_setpoint",
        pv_for_selected_magnet="mux_sel_selected",
        read_from_file=True,
    )
    # measurement was made using power converter names: we work switching power converters
    # but now it is per magnet: so the name should be magnet names
    preprocessed_measurement = MeasurementData(
        measurement=[
            MeasurementPerMagnet(per_magnet=m.per_magnet, name=m.name.replace("P", "M"))
            for m in tqdm.tqdm(
                preprocessed_measurement.measurement,
                total=len(preprocessed_measurement.measurement),
                desc="rename: power converter to magnet   ",
            )
        ]
    )
    # What a hack ... need to get a consistent source
    element_families.add_element_names(
        [datum.name for datum in preprocessed_measurement.measurement]
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

    name_pos_service = DeviceLocationServiceBESSYII()
    h_st = list(element_families.get(BESSYIIFamilyNames.horizontal_steerers).members)
    v_st = list(element_families.get(BESSYIIFamilyNames.vertical_steerers).members)

    def name_to_position(device_name: str) -> float:
        return model.survey.get(
            name_pos_service.get_location_name(device_name)
        ).value

    h_st.sort(key=name_to_position)
    v_st.sort(key=name_to_position)

    orms = extract_response_matrices_per_steerers(
        fit_results, horizontal_steerer_names=h_st, vertical_steerer_names=v_st)
    pv_plot_orms(orms, model, name_pos_service=name_pos_service, scale_bpm_readings=1e-2)

    try:
        mpl_plot_orms(orms)
    except Exception:
        raise
    else:
        plt.show()
