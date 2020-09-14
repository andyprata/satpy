# This is ahi_l1b_nc.py

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample.geometry import create_area_def
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE
from satpy.readers.utils import get_earth_radius


logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'H08': 'Himawari-8',
    'H09': 'Himawari-9',
}

NC_VIS_DATASETS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06']
NC_IR_DATASETS = ['B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15']
NC_GEOMETRY_DATASETS = ['SAZ', 'SAA', 'SOZ', 'SOA']


class NCAHIFileHandler(BaseFileHandler):
    """Reader for AHI L1B NetCDF4 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(NCAHIFileHandler, self).__init__(filename, filename_info, filetype_info)
        # xarray's default netcdf4 engine
        try:
            self.nc = xr.open_dataset(self.filename, chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE})
        except ValueError:
            self.nc = xr.open_dataset(self.filename, chunks={'longitude': CHUNK_SIZE, 'latitude': CHUNK_SIZE})

        platform_shortname = filename_info['platform_shortname']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)

    @property
    def start_time(self):
        # Reference: https://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
        return datetime.utcfromtimestamp(self.nc.start_time.values[0].astype('O')/1e9)

    @property
    def end_time(self):
        # Reference: https://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
        return datetime.utcfromtimestamp(self.nc.end_time.values[0].astype('O')/1e9)

    def get_dataset(self, key, info):
        """Get the dataset."""
        logger.debug('Reading %s.', key['name'])

        if key['name'] in NC_GEOMETRY_DATASETS:
            data = self.nc[key['name']]
        else:
            data = self.read_band(key, info)

        # Rename dimensions to correspond to satpy's 'y' and 'x' standard.
        if 'longitude' in data.dims:
            data = data.rename({'longitude': 'x'})
        if 'latitude' in data.dims:
            data = data.rename({'latitude': 'y'})
        return data

    def read_band(self, key, info):
        if key['name'] in NC_VIS_DATASETS:
            nc_band_name = key['name'].replace('B', 'albedo_')
            # Convert albedo to reflectance in %
            data = (self.nc[nc_band_name] / np.cos(self.nc['SOZ'] * np.pi / 180.)) * 100.
        elif key['name'] in NC_IR_DATASETS:
            nc_band_name = key['name'].replace('B', 'tbb_')
            data = self.nc[nc_band_name]
        else:
            raise ValueError('Not a valid dataset!')

        # Read in the satellite geometry parameters as a list of strings.
        geometry_parameters = np.array(self.nc.geometry_parameters.long_name.split(','), dtype=str)
        sub_lon = float(self.nc.geometry_parameters.values[geometry_parameters == 'sub_lon'])
        # Get actual satellite position. For altitude use the ellipsoid radius at the SSP.
        actual_lon = float(self.nc.geometry_parameters.values[geometry_parameters == 'SSPlon']) * 180./np.pi
        actual_lat = float(self.nc.geometry_parameters.values[geometry_parameters == 'SSPlat']) * 180./np.pi
        re = get_earth_radius(lon=actual_lon, lat=actual_lat,
                              a=float(self.nc.geometry_parameters.values[geometry_parameters == 'req'] * 1000),
                              b=float(self.nc.geometry_parameters.values[geometry_parameters == 'rpol'] * 1000))
        actual_alt = float(self.nc.geometry_parameters.values[geometry_parameters == 'Rs'] * 1000) - re

        # Update metadata
        new_info = dict(
            units=info['units'],
            standard_name=info['standard_name'],
            wavelength=info['wavelength'],
            resolution='resolution',
            id=key,
            name=key['name'],
            #scheduled_time=self.scheduled_time,
            platform_name=self.platform_name,
            sensor=info['sensor'],
            satellite_longitude=actual_lon,
            satellite_latitude=actual_lat,
            satellite_altitude=actual_alt,
            orbital_parameters={
                'projection_longitude': sub_lon,
                'projection_latitude': 0.,
                'projection_altitude': actual_alt,
                'satellite_actual_longitude': actual_lon,
                'satellite_actual_latitude': actual_lat,
                'satellite_actual_altitude': actual_alt,
                #'nadir_longitude': float(self.nav_info['nadir_longitude']),
                #'nadir_latitude': float(self.nav_info['nadir_latitude'])
            }
        )
        data.attrs = new_info
        return data

    def get_area_def(self, key):
        """Get the area definition."""
        res = 0.02  # grid resolution in degrees
        extent = [80, 200, -60, 60]
        ahi_l1b_nc_def = create_area_def('Himawari-8 AHI equal latitude-longitude grid',
                                         {'proj': 'latlong', 'datum': 'WGS84'},
                                         # Need to expand corners of grid so that extent boundaries fall in grid box centres
                                         area_extent=[extent[0] - res / 2., extent[2] - res / 2.,
                                                      extent[1] + res / 2., extent[3] + res / 2.],
                                         resolution=res,
                                         units='degrees',
                                         description='Himawari-8 AHI equal latitude-longitude grid')
        return ahi_l1b_nc_def

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        try:
            self.nc.close()
        except (IOError, OSError, AttributeError):
            pass
