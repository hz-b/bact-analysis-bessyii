"""
So that repeated data loading is faster ...
"""

from databroker import catalog
import xarray as xr
from .preprocess_data import load_and_check_data
from bact_analysis.utils.preprocess import reorder_by_groups
import functools
import logging

logger = logging.getLogger('bact-analysis-bessyii')

@functools.lru_cache
def load_and_rearrange_data(uid: str, catalog_name: str = "heavy_local") -> xr.Dataset:
    """load data using uid and make it selectable per magnet
    Todo:
        Require loading from other formats?
    """
    try:
        db = catalog[catalog_name]
        run = db[uid]
    except:
        logger.warning(f'using catalog name {catalog_name} uid {uid}')
        raise
    return load_and_check_data(run, device_name="bpm")

__all__ = ["load_and_rearrange_data"]
