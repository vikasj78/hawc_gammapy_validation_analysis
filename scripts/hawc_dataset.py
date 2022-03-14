import matplotlib.pyplot as plt
from gammapy.data import DataStore, HDUIndexTable, ObservationTable
from gammapy.irf import EDispKernelMap,PSFMap
from gammapy.datasets import MapDataset
from gammapy.maps import WcsGeom, MapAxis, Map,HpxGeom, WcsNDMap , HpxNDMap
from gammapy.makers import MapDatasetMaker
from gammapy.data import Observation
from astropy.table import Table
from astropy.time import Time
from gammapy.data import EventList, GTI,Observation
import astropy.units as u
from astropy.coordinates import SkyCoord
import glob
import numpy as np
from astropy.io import fits
import pytest
import math
import matplotlib as mpl
import healpy as hp
import yaml
import logging
import argparse
from pathlib import Path

from hawc_dataset import HAWCMapDataset

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def read_hawc_hpx_map(filename, field=0):
    """Read HAWC maps and create a HpxMap object.
    
    TODO: this could be supported by Gammapy as HpxNDMap.read(filename, format="hawc", field=)
    
    Parameters
    ----------
    filename : str
        Input map filename
    field : int
        Field number

    Returns
    -------
    hpx_map : HpxNDMap
        Gammapy healpix map
    """
    log.info("Reading {}".format(filename))

    # TODO: creating an allsky map first and extracting only a small part is inefficient
    hpx_map = HpxNDMap.create(nside=1024, nest=False, frame="icrs")
    hpx_map.data = hp.read_map(filename, field=field, verbose=False)
    
    # Create a partial HEALPix geometry
    coords = HPX_PARTIAL_GEOM_IMAGE.get_coord()
    data = hpx_map.get_by_coord(coords)
    return HpxNDMap.from_geom(geom=HPX_PARTIAL_GEOM_IMAGE, data=data)


def get_hpx_cube(filenames, field, unit=""):
    """Get healpix cube with "fake energy" axis.
    
    Parameters
    ----------
    filenames : list of str
        List of input files
    field : int
        Field number
    unit : str
        Unit of the map data

    Returns
    -------
    hpx_map : HpxNDMap
        Gammapy healpix cube
    """
    map_hpx = HpxNDMap.from_geom(geom=HPX_PARTIAL_GEOM, unit=unit)

    for idx, filename in enumerate(filenames):
        map_hpx_image = read_hawc_hpx_map(filename, field=field)
        map_hpx.data[idx] = map_hpx_image.data

    return map_hpx

def get_counts_map(filenames):
    counts_hpx = get_hpx_cube(filenames, field=0, unit="")
    return counts_hpx


def get_background_map(filenames):
    background_hpx = get_hpx_cube(filenames, field=1, unit="")
    background_hpx.data /= background_hpx.geom.solid_angle().to_value("deg2")

    # reproject exposure and background to local WCS using a TAN projection
    background = Map.from_geom(WCS_GEOM)
    coords = WCS_GEOM.get_coord()
    background.data = background_hpx.get_by_coord(coords)
    background.data *= WCS_GEOM.solid_angle().to_value("deg2")
    return background



IRF_path = "/lfs/l7/hawc/hawc-gammapy-verification/IRFs/"
exposure_path = "/lfs/l7/hawc/hawc-gammapy-verification/point_source/crab_effective_exposure/"



def write_hawc_dataset(filenames, nHitBin=0):
    counts_hpx = get_counts_map(filenames)
#    counts_hpx.write("/lfs/l7/hawc/hawc-gammapy-verification/point_source/crab_events_dataset/counts_maps/hpx-hawc-map-counts-{}nHitBin.fits.gz".format(nHitBin))
    background = get_background_map(filenames)
#    background.write("/lfs/l7/hawc/hawc-gammapy-verification/point_source/crab_events_dataset/background_maps/nan-hawc-map-bg-{}nHitBin.fits.gz".format(nHitBin))

    # create mask to exclude non valid data
    mask_safe = HpxNDMap.from_geom(geom=counts_hpx.geom)
    mask_safe.data = ~np.isnan(counts_hpx.data)

    # Get the energy true from one of the IRFs: I want it to be exactly the same
    effective_arearef = Map.read(exposure_path + 'EffectiveExposureMap_bin1GP_energy.fits')
    energy_true = effective_arearef.geom.axes[0]

    # Create the dataset with that geometry, energy axis true and name
    dataset = HAWCMapDataset.create(geom=WCS_GEOM, name="nHit-" + str(nHitBin),energy_axis_true=energy_true)

    # Initialize the maker with the selection that I want (no psf)
    maker = MapDatasetMaker(selection = ["exposure", "edisp", "psf"])

    # For each bin, I define a HDUIndexTable, which is just a table that has, for each "observation", which in
    # your case is just one, the path to the relevant files: the effective exposure (=effectiveArea*nr transits in each RA),
    # the edisp, the psf...
    obs_table = ObservationTable(names = ['OBS_ID'],dtype=['str'])
    crabpass4 = HDUIndexTable(names= ['OBS_ID',
         'HDU_TYPE',
         'HDU_CLASS',
         'FILE_DIR',
         'FILE_NAME',
         'HDU_NAME'],dtype=6*['str'])

    # For each bin, I define an ObservationTable, which is just a collection of names for
    # the different observations: in my case the chunks, in yours just the one Map per nhit bin
    obs_table.add_row(vals=[str(nHitBin)])
    # And to the HDUIndexTable the corresponding files. Note that I skip the PSF and the background.
    # Each table entry has the observationID (str(nHitBin)), the name of which component I
    # am adding ("aeff", "edisp"), the class of that object in gammapy ("map", "edisp_kernel_map"), and the folder and filename.

    # aeff
    aeff_row = [str(nHitBin), 'aeff','map', exposure_path, 'EffectiveExposureMap_bin'+str(nHitBin)+'GP_energy.fits','aeff']
    crabpass4.add_row(vals = aeff_row)

    # edisp
    edisp_row = [str(nHitBin), 'edisp','edisp_kernel_map', IRF_path, 'EDispKernelMap_bin'+str(nHitBin)+'_energy.fits','edisp']
    crabpass4.add_row(vals = edisp_row)

    # psf
    psf_row = [str(nHitBin), 'psf','psf_map', IRF_path, 'PSFMap_bin'+str(nHitBin)+'_energy.fits','psf']
    crabpass4.add_row(vals = psf_row)

    data_store = DataStore(obs_table=obs_table, hdu_table=crabpass4)
    observations = data_store.get_observations()        

    # run it
    dataset = maker.run(dataset, observations[0])


    # In principle you should be able to run the PSF through the maker, but what the Maker returns is the PSFMap, not the PSFKernel,
    # so to avoid problems with the enery axis I'll just extract it
    dataset.psf = dataset.psf.get_psf_kernel(position=SkyCoord(SRC_RA, SRC_DEC, unit='deg'), geom=WCS_GEOM.as_energy_true, max_radius="3 deg", factor=1)

    dataset.counts = counts_hpx
    dataset.background = background
    dataset.mask_safe = mask_safe

    filename = Path("gammapy_dataset_vikas") / "TESTcrab-{}nHitBin.fits.gz".format(nHitBin)
    log.info("Writing {}".format(filename))
    dataset.write(filename, overwrite=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="source specifications")
    p.add_argument("--ra", dest="ra", nargs="?", help="RA in deg", default=83.63)
    p.add_argument("--dec", dest="dec", nargs="?", help="Dec in deg", default=22.01)
    p.add_argument("--roi", dest="roi", nargs="?", help="ROI in deg", default=3.0)
    #source specification

    args = p.parse_args()
    SRC_RA = args.ra #deg
    SRC_DEC = args.dec #deg
    SRC_ROI = args.roi #deg

    center_dec_bins = np.arange(-90,90,5)
    dec_bin = center_dec_bins[(np.abs(center_dec_bins - SRC_DEC)).argmin()]
    #analysis Bins
    BIN_SETS = []
    BINS_ALL_DEC = yaml.load(open("/lfs/l7/hawc/hawc-gammapy-verification/hawc_resources/data/crab/hawc_bins.yml",'r'))
    BINS = np.array(BINS_ALL_DEC[dec_bin].split(' ')) #20 is the nearest declination to crab for now hard-coded 

    nHitBins = np.arange(1,10).astype(str)
    for nh in nHitBins:
        BIN_SETS.append(list(BINS[np.flatnonzero(np.core.defchararray.find(BINS,nh)!=-1)]))

    print(BIN_SETS)
    TOTAL_ENERGY_AXIS = MapAxis.from_edges(
        [0.316,0.562,1.00,1.78,3.16,5.62,10.0,17.8,31.6,56.2,100,177,316] * u.TeV,
        name="energy",
        interp="log"
    )
    ENERGY_BINS = ['a','b','c','d','e','f','g','h','i','j','k','l']
    for nHitBin,USE_BINS in enumerate(BIN_SETS):
        nHitBin += 1
        firstEnergyBin = ENERGY_BINS.index(USE_BINS[0][-1])
        lastEnergyBin = ENERGY_BINS.index(USE_BINS[-1][-1])
        # energy axis
        ENERGY_AXIS = TOTAL_ENERGY_AXIS.slice(slice(firstEnergyBin, lastEnergyBin + 1))
        print('Event class:',nHitBin, 'Energy edges:', ENERGY_AXIS.edges)
        ENERGY_AXIS_TRUE = MapAxis.from_bounds(1e-3, 10e3, nbin=len(USE_BINS), name="energy_true", interp="log", unit="TeV")

        # HEALPix geometries
        REGION = 'DISK({},{},{})'.format(SRC_RA,SRC_DEC,SRC_ROI)
        print(REGION)
        HPX_PARTIAL_GEOM_IMAGE = HpxGeom(nside=1024, region=REGION, nest=False)
        HPX_PARTIAL_GEOM = HPX_PARTIAL_GEOM_IMAGE.to_cube([ENERGY_AXIS])
        HPX_PARTIAL_GEOM_TRUE = HPX_PARTIAL_GEOM_IMAGE.to_cube([ENERGY_AXIS_TRUE])

        # local WCS geometries
        WCS_GEOM = HPX_PARTIAL_GEOM.to_wcs_geom(drop_axes=False, oversample=1, proj="TAN")
        WCS_GEOM_TRUE = HPX_PARTIAL_GEOM_TRUE.to_wcs_geom(drop_axes=False, oversample=1, proj="TAN")
        path = Path("/lfs/l7/hawc/hawc-gammapy-verification/hawc_resources/data/crab")
        filenames = []
        for b in USE_BINS:
            filenames.append(path / "map_bin{}.fits.gz".format(b))
        write_hawc_dataset(filenames, nHitBin=nHitBin)

