import numpy as np

from .bessyii_lattice import twiss_from_at
from ..tools.bpm_repo import BPMCalibrationsRepositoryBESSYII
from ..tools.clean_data import fit_ready_data_per_magnet_clip_steps
from ..tools.correct_bpm_naming import measurement_per_magnet_bpm_data_correct_name
from matplotlib import pyplot as plt

from .bessyii_info_repos import (
    SpaceMappingCollectionBESSYII,
    DeviceLocationServiceBESSYII,
    BessyIIELementFamilies,
)
from .measured_data_cleaning import measurement_data_with_known_bpms_only
from ..model.accelerator_model import (
    AcceleratorDescription,
    SurveyPositions,
    Position
)
from ..model.orbit_prediciton import OrbitPredictionCollection
from .plot_matplotlib import plot_bpm_offsets, plot_forecast_difference
from .plot_pyvista import plot_forecast_difference_3d
from .prepare_plot_data import compute_prediction_per_magnet
from .steerer_excitations import (
    fit_steerer_response_one_separate_per_plane,
    fit_steerer_response_one_both_planes,
)

from ..model.analysis_model import (
    FitReadyData,
    MeasurementData,
    EstimatedAngles,
    MeasurementPerMagnet,
)
from ..model.analysis_util import (
    flatten_for_fit,
    measurement_per_magnet_bpms_raw_data_to_m,
)
from ..tools.preprocess_data import load_and_rearrange_data
import tqdm


def main(uid, n_magnets=None):
    model = twiss_from_at()
    space_col = SpaceMappingCollectionBESSYII()
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
                desc="rename bpm raw   ",
            )
        ]
    )

    if False:
        # correct the bpm names
        preprocessed_measurement = MeasurementData(
            measurement=[
                measurement_per_magnet_bpm_data_correct_name(m)
                for m in tqdm.tqdm(
                    preprocessed_measurement.measurement,
                    desc="rename bpm raw   ",
                )
            ]
        )
    preprocessed_measurement = MeasurementData(
        measurement=[
            measurement_per_magnet_bpms_raw_data_to_m(m, calib_repo=BPMCalibrationsRepositoryBESSYII())
            for m in tqdm.tqdm(
                preprocessed_measurement.measurement,
                desc="conv bpm raw -> m",
            )
        ]
    )

    # reduce ourselves to the set of bpms that are known to the
    # model and the ring
    bpm_names_of_model = [name for name in model.twiss.names if name[:3] == "BPM"]
    bpm_names_of_measurement = [
        bpm["name"] for bpm in preprocessed_measurement.measurement[0].per_magnet[0].bpm
    ]
    # could use set intersection: I am not sure since when the
    # set guarantees that the element order is conserved
    # I'd like to keep the elements the way they are in the model
    # as these are presumably in a consecutive order along the ring
    bpm_names_known_model_measurement = [
        bpm_name
        for bpm_name in bpm_names_of_model
        if bpm_name in bpm_names_of_measurement
    ]
    preprocessed_measurement = measurement_data_with_known_bpms_only(
        preprocessed_measurement, bpm_names_known_model_measurement
    )

    # What a hack ... need to get a consistent source
    element_families.add_element_names(
        [datum.name.replace("P", "M") for datum in preprocessed_measurement.measurement]
    )

    fit_ready_data = FitReadyData(
        per_magnet=[
            flatten_for_fit(
                measurement.per_magnet, measurement.name, pos="pos", rms="rms"
            )
            for measurement in tqdm.tqdm(
                preprocessed_measurement.measurement,
                desc="rearrange  data for fit: ",
            )
        ]
    )

    # remove first reading as this is wrong ...
    # Todo: need to remove as soon as data taking is corrected
    steps = set([step for step in fit_ready_data.per_magnet[0].steps if step > 0])
    # remove first entries ... need to fix
    fit_ready_data = FitReadyData(
        per_magnet=[
            fit_ready_data_per_magnet_clip_steps(for_magnet, steps)
            for for_magnet in tqdm.tqdm(
                fit_ready_data.per_magnet,
                desc="only allowed steps",
            )
        ]
    )

    name_pos_service = DeviceLocationServiceBESSYII()
    steerer_response_fit_2d = EstimatedAngles(
        per_magnet=[
            fit_steerer_response_one_both_planes(
                data,
                name_pos_service.get_location_name(data.name),
                space_col=space_col,
                model=model.twiss,
            )
            for data in tqdm.tqdm(
                fit_ready_data.per_magnet,
                desc="fitting steerer excitations (both planes)"
            )
        ],
        md=None,
    )

    steerer_response_fit = EstimatedAngles(
        per_magnet=[
            fit_steerer_response_one_separate_per_plane(
                data,
                name_pos_service.get_location_name(data.name),
                space_col=space_col,
                model=model.twiss,
            )
            for data in tqdm.tqdm(
                fit_ready_data.per_magnet,
                desc="fitting steerer excitations",
            )
        ],
        md=None,
    )

    orbit_prediction = OrbitPredictionCollection(
        per_magnet=[
            compute_prediction_per_magnet(
                kick_values_from_fit,
                measurement.excitations,
                pos_names_for_measurement=bpm_names_known_model_measurement,
                magnet_name=measurement.name,
            )
            for measurement, kick_values_from_fit in
                zip(tqdm.tqdm(fit_ready_data.per_magnet, desc="compute forecast",), steerer_response_fit.per_magnet)
        ]
    )

    # plot_steerer_response(fit_ready_data, orbit_prediction)
    plot_forecast_difference_3d(
        fit_ready_data,
        orbit_prediction,
        model,
        element_families,
        name_pos_service=name_pos_service,
    )
    try:
        plt.ion()
        plot_forecast_difference(
            fit_ready_data, orbit_prediction, steerer_response_fit_2d, model
        )
        plot_bpm_offsets(fit_ready_data, steerer_response_fit_2d, model)
    except Exception:
        plt.ioff()
        raise
    else:
        plt.ioff()
        plt.show()
