import re

from bact_device_models.filters.bpm_calibration import BPMCalibration, BPMCalibrationPlane
from ..model.analysis_util import BPMCalibrationsRepository
from bact_bessyii_ophyd.devices.pp.bpm_parameters import create_bpm_config
import functools
import logging

logger = logging.getLogger("bact-analysis-bessyii")

bpm_config = create_bpm_config()

class BPMCalibrationsRepositoryBESSYII(BPMCalibrationsRepository):
    def __init__(self):
        self.bpm_calibrations_default = BPMCalibration(
            x=BPMCalibrationPlane(), y=BPMCalibrationPlane()
        )
        self.bpm_calibrations_inactive = BPMCalibration(
                x=BPMCalibrationPlane(active=False),
                y=BPMCalibrationPlane(active=False),
        )
        self.bpm_calibrations = dict(
            BPMZ4D2R=BPMCalibration(
                x=BPMCalibrationPlane(scale=0.3e-3), y=BPMCalibrationPlane()
            ),
            BPMZ41T6R=BPMCalibration(
                x=BPMCalibrationPlane(scale=0.2489622e-3),
                y=BPMCalibrationPlane(scale=0.8050295e-3),
            ),
        )


    @functools.lru_cache(maxsize=None)
    def get(self, name):
        r = self.bpm_calibrations_default
        try:
            r = self.bpm_calibrations[name]
        except KeyError:
            logger.info("bpm with name %s uses (presumably) default configuration", name)
            pass

        if name not in bpm_config['name']:
            logger.warning("bpm with name %s (persumably) inactive", name)
            r = self.bpm_calibrations_inactive
        assert(r is not  None)
        return r
