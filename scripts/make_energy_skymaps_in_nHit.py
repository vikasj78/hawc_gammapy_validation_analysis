#!/usr/bin/env python
"""Convert HAWC maps to Gammapy format"""
import logging
from pathlib import Path
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits
from astropy import units as u
from astropy.time import Time

import healpy as hp
from gammapy.maps import MapAxis, HpxNDMap, HpxGeom, Map
from gammapy.irf import EDispKernel, EnergyDependentTablePSF, PSFKernel
from pandas import HDFStore

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


def get_table_psf(store, dec_center=22):
    df_aeff = store['/effective_area']
    df_psf = store['/psf']

    #energy_bins = df_aeff.index.levels[1]
    energy_bins = USE_BINS
    psf_data = np.empty(shape=(len(energy_bins), 500))

    for idx_energy, energy_bin in enumerate(energy_bins):
        psf_values = df_psf.loc[dec_center, energy_bin, :]

        y = psf_values["ys"].values
        x = psf_values["xs"].values

        p = np.pi * y / x
        psf_data[idx_energy] = p

    rad = x * u.deg

    table_psf = EnergyDependentTablePSF(
        psf_value=psf_data * u.Unit("deg-2"),
        energy=ENERGY_AXIS.center,
        rad=rad
    )

    containment = table_psf.containment(table_psf.energy, 15 * u.deg)
    table_psf.psf_value /= containment

    # delete cached quantitites
    del table_psf.__dict__["_interpolate_containment"]
    del table_psf.__dict__["_interpolate"]
    return table_psf


def get_edisp_kernel(store, energy_axis, dec_center=22):
    """Get enerfgy dispersion matrix

    Parameters
    ----------
    store : HDFStore
        HDF store with response data
    energy_axis : MapAxis
        "fake" energy axis
    dec_center : float
        Declination where response is extracted.
    
    Returns
    -------
    edisp : EDispKernel
        Energy dispersion matrix
    """
    df_aeff = store['/effective_area']
    #energy_bins = df_aeff.index.levels[1]
    energy_bins = USE_BINS
    edisp_data = np.empty(shape=(140, len(energy_bins)))

    for idx_energy, energy_bin in enumerate(energy_bins):
        aeff_values = df_aeff.loc[dec_center, energy_bin, :]
        edisp_values = aeff_values["sim_signal_events_per_bin"]
        edisp_values = gaussian_filter(edisp_values, 2)
        edisp_data[:, idx_energy] = edisp_values

    e_true_lo = aeff_values["sim_energy_bin_low"].values * u.TeV
    e_true_hi = aeff_values["sim_energy_bin_hi"].values * u.TeV
    e_reco_lo = energy_axis.edges[:-1]
    e_reco_hi = energy_axis.edges[1:]

    return EDispKernel(
        e_true_lo=e_true_lo,
        e_true_hi=e_true_hi,
        e_reco_lo=e_reco_lo,
        e_reco_hi=e_reco_hi,
        data=edisp_data,
    )


def get_exposure_map(store, geom_image, dec_center=20):
    """Get exposure map.

    Parameters
    ----------
    store : `~pandas.HDFStore`
        Hdf5 store.
    geom_image : `~gammapy.WcsGeom`
        Spatial geometry of the exposure map. The energy info is taken
        from the response file.
    dec_center : float
        Declination where response is extracted.

    Returns
    -------
    exposure : `WcsNDMap`
        Exposure map
    """

    hdulist = fits.open(Path("/lfs/l2/hawc/users/vikasj78/energy-skymaps/kelly/crab_maps") / "energy_skymaps_gp_crab_bin1a.fits.gz")

    t_start = Time(hdulist[0].header["STARTMJD"], format="mjd")
    t_stop = Time(hdulist[0].header["STOPMJD"], format="mjd")

    factor = ((t_stop - t_start) / u.sday).to("")

    df_aeff = store['/effective_area']
    #energy_bins = df_aeff.index.levels[1]
    energy_bins = USE_BINS
    aeff_values = df_aeff.loc[dec_center, energy_bins[0], :]

    e_min = aeff_values["sim_energy_bin_low"].values * u.TeV
    e_max = aeff_values["sim_energy_bin_hi"].values * u.TeV
    e_ref = aeff_values["sim_energy_bin_centers"].values * u.TeV
    dnde = (aeff_values["sim_differential_photon_fluxes"].values * u.Unit("cm-2 s-1 TeV-1"))

    data = factor / ((e_max - e_min) * dnde)

    energy_axis_true = MapAxis.from_nodes(e_ref, name="energy", interp="log")
    geom = geom_image.to_cube([energy_axis_true])

    exposure = Map.from_geom(geom, unit=data.unit)
    exposure.quantity += data[:, np.newaxis, np.newaxis]
    return exposure


def write_hawc_dataset(filenames, filename_response, dec_center=20, nHitBin=0):
    """Convert HAWC maps into the gadf for ND maps.

    Parameters
    ----------
    filenames : list of str
        Filenames.
    filename_reponse : str
        Filename of the response file
    dec_center : float
        Declination where response is extracted.

    """
    counts_hpx = get_counts_map(filenames)
    background = get_background_map(filenames)

    log.info(f"Reading: {filename_response}")
    store = HDFStore(filename_response)

    geom_image = WCS_GEOM.to_image()
    exposure = get_exposure_map(store, geom_image=geom_image, dec_center=dec_center)
    table_psf = get_table_psf(store, dec_center=dec_center)
    print(table_psf)
    # TODO: writing the table psf is currently broken, but only needed for visualisation
    #filename = Path("gp-hawc-maps") / "hawc-table-psf.fits.gz"
    #log.info("Writing {}".format(filename))
    #table_psf.write(filename, overwrite=True)

    psf = PSFKernel.from_table_psf(table_psf, geom_image.to_cube([ENERGY_AXIS_TRUE]), max_radius="2 deg")
    print(ENERGY_AXIS)
    edisp = get_edisp_kernel(store, ENERGY_AXIS, dec_center=dec_center)

    # create mask to exclude non valid data
    mask_safe = HpxNDMap.from_geom(geom=counts_hpx.geom)
    mask_safe.data = ~np.isnan(counts_hpx.data)
    
    exclude_primary = slice(1, None)

    hdu_primary = fits.PrimaryHDU()
    hdulist = fits.HDUList([hdu_primary])
            
    hdulist += counts_hpx.to_hdulist(hdu="counts")[exclude_primary]
    hdulist += exposure.to_hdulist(hdu="exposure")[exclude_primary]
    hdulist += background.to_hdulist(hdu="background")[exclude_primary]
    hdulist += psf.psf_kernel_map.to_hdulist(hdu="psf_kernel")[exclude_primary]

    hdus = edisp.to_hdulist()
    hdus["MATRIX"].name = "edisp_matrix"
    hdus["EBOUNDS"].name = "edisp_matrix_ebounds"
    hdulist.append(hdus["EDISP_MATRIX"])
    hdulist.append(hdus["EDISP_MATRIX_EBOUNDS"])

    mask_safe_int = mask_safe.copy()
    mask_safe_int.data = mask_safe_int.data.astype(int)
    hdulist += mask_safe_int.to_hdulist(hdu="mask_safe")[exclude_primary]
    
    filename = Path("/lfs/l2/hawc/users/vikasj78/energy-skymaps/kelly/crab_maps") / "hawc-energy-skymaps-gammapy-{}nHitBin.fits.gz".format(nHitBin)
    log.info("Writing {}".format(filename))
    hdulist.writeto(filename, overwrite=True)


if __name__ == "__main__":
    #analysis Bins
    BIN_SETS = [['1a', '1b', '1c', '1d', '1e', '1f', '1g'], ['2a', '2b', '2c', '2d', '2e', '2f'], ['3b',
                '3c', '3d', '3e', '3f', '3g'], ['4c', '4d', '4e', '4f', '4g', '4h'], ['5c', '5d', '5e',
                '5f', '5g', '5h'], ['6d', '6e', '6f', '6g', '6h', '6i'], ['7e', '7f', '7g', '7h', '7i',
                '7j'], ['8f', '8g', '8h', '8i', '8j'], ['9g', '9h', '9i', '9j', '9k', '9l']] 

    for nHitBin,USE_BINS in enumerate(BIN_SETS):
        # configuration of fake energy axis
        ENERGY_AXIS = MapAxis.from_bounds(0.1, 1e3, nbin=len(USE_BINS), name="energy", interp="log", unit="TeV")
        ENERGY_AXIS_TRUE = MapAxis.from_bounds(0.1, 1e3, nbin=len(USE_BINS), name="energy_true", interp="log", unit="TeV")

        # HEALPix geometries
        HPX_PARTIAL_GEOM_IMAGE = HpxGeom(nside=1024, region='DISK(83.63,22.01,2.0)', nest=False)
        HPX_PARTIAL_GEOM = HPX_PARTIAL_GEOM_IMAGE.to_cube([ENERGY_AXIS])
        HPX_PARTIAL_GEOM_TRUE = HPX_PARTIAL_GEOM_IMAGE.to_cube([ENERGY_AXIS_TRUE])

        # local WCS geometries
        WCS_GEOM = HPX_PARTIAL_GEOM.make_wcs(drop_axes=False, oversample=1, proj="TAN")
        WCS_GEOM_TRUE = HPX_PARTIAL_GEOM_TRUE.make_wcs(drop_axes=False, oversample=1, proj="TAN")
        path = Path("/lfs/l2/hawc/users/vikasj78/energy-skymaps/kelly/crab_maps")
        filenames = []
        for b in USE_BINS:
            filenames.append(path / "energy_skymaps_gp_crab_bin{}.fits.gz".format(b))
        filename_response = path / "crab_response.hd5"
        nHitBin = nHitBin + 1
        write_hawc_dataset(filenames, filename_response=filename_response, dec_center=20, nHitBin=nHitBin)
