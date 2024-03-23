from typing import Sequence

from bact_analysis_bessyii.model.analysis_model import MeasurementData, MeasurementPerMagnet, MeasurementPoint


def measurement_data_with_known_bpms_only(data: MeasurementData, known_bpm_names: Sequence[str]):
    return MeasurementData(
        measurement=[
            measurement_data_per_magnet_with_known_bpms_only(measurement, known_bpm_names)
            for measurement in data.measurement
        ]
    )


def measurement_data_per_magnet_with_known_bpms_only(data: MeasurementPerMagnet, known_bpm_names: Sequence[str]) -> MeasurementPerMagnet:
    return MeasurementPerMagnet(
        name=data.name,
        per_magnet=[measurement_point_with_known_bpms_only(
            measurement_point, known_bpm_names=known_bpm_names
        )
        for measurement_point in data.per_magnet])

def measurement_point_with_known_bpms_only(measurement_point: MeasurementPoint, known_bpm_names: Sequence[str]) -> MeasurementPoint:
    return MeasurementPoint(
        step=measurement_point.step,
        excitation=measurement_point.excitation,
        bpm=[bpm_data for bpm_data in measurement_point.bpm if bpm_data["name"] in known_bpm_names]
    )


__all__ = ["measurement_data_with_known_bpms_only"]