from ..model.analysis_model import BPMCalibration, BPMCalibrationPlane
from ..model.analysis_util import BPMCalibrationsRepository
import functools


_inactive_bpms = (
    "BPMZ8T2R",
    "BPMZ8T4R",
    "BPMZ8D5R",
    "BPMZ8T7R",
    "BPMZ8D8R",
    "BPMZ43D1R",
    "BPMZ44D1R",
    "BPMZ43T1R",
    "BPMZ2T3R",
    "BPMZ41T3R",
    "BPMZ42T3R",
    "BPMZ7D4R",
    "BPMZ41D6R",
    "BPMZ42D6R",
    "BPMZ43D6R",
    "BPMZ44D6R",
    "BPMZ41D1R",
    "BPMZ42D1R",
    "BPMZ5D1R",
)


class BPMCalibrationsRepositoryBESSYII(BPMCalibrationsRepository):
    def __init__(self):
        self.bpm_calibrations = dict(
            BPMZ4D2R=BPMCalibration(
                x=BPMCalibrationPlane(scale=0.3e-3), y=BPMCalibrationPlane()
            ),
            BPMZ41T6R=BPMCalibration(
                x=BPMCalibrationPlane(scale=0.2489622e-3),
                y=BPMCalibrationPlane(scale=0.8050295e-3),
            ),
        )
        for name in _inactive_bpms:
            self.bpm_calibrations[name] = BPMCalibration(
                x=BPMCalibrationPlane(active=False),
                y=BPMCalibrationPlane(active=False),
            )

        self.bpm_calibrations_default = BPMCalibration(
            x=BPMCalibrationPlane(), y=BPMCalibrationPlane()
        )

    @functools.lru_cache(maxsize=None)
    def get(self, name):
        r = self.bpm_calibrations_default
        try:
            r = self.bpm_calibrations[name]
        except KeyError:
            logger.info("bpm with name %s uses default configuration", name)
            pass
        return r
