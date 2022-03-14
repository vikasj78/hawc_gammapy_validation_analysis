import sys, os
import pandas as pd
import numpy as np
def galactic_to_radec(l, b):
    ''' convert from galactic coordinates to J2000 coordinates. '''
    from astropy.coordinates import ICRS, Galactic
    from astropy.units import deg
    c = Galactic(l=l*deg, b=b*deg).transform_to(ICRS)
    return c.ra.value, c.dec.value

def angle_between(phi1, theta1, phi2, theta2, unit='deg'):
    phi1 = np.asarray(phi1, dtype='float').copy()
    phi2 = np.asarray(phi2, dtype='float').copy()
    theta1 = np.asarray(theta1, dtype='float').copy()
    theta2 = np.asarray(theta2, dtype='float').copy()
    if unit == 'deg':
        phi1 *= np.pi / 180
        phi2 *= np.pi / 180
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180
    ax1 = np.cos(phi1)  * np.cos(theta1)
    ay1 = np.sin(-phi1) * np.cos(theta1)
    az1 = np.sin(theta1)
    ax2 = np.cos(phi2)  * np.cos(theta2)
    ay2 = np.sin(-phi2) * np.cos(theta2)
    az2 = np.sin(theta2)
    res = np.arccos(np.clip(ax1*ax2 + ay1*ay2 + az1*az2, -1, 1))
    if unit == 'deg':
        return res * 180 / np.pi
    return res

class ExclusionRegionSet(list):
    def __init__(self, regions=[]):
        for r in regions:
            if not isinstance(r, ExclusionRegion):
                raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.__init__(self, regions)

    def append(self, region):
        if not isinstance(region, ExclusionRegion):
            raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.append(self, region)

    def extend(self, regions):
        for r in regions:
            if not isinstance(r, ExclusionRegion):
                raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.extend(self, regions)

    @classmethod
    def from_file(cls, filename):
        ers = cls()
        ers.read_from_file(filename)
        return ers

    def read_from_file(self, filename):
        dat = pd.read_csv(filename, comment='#',
                                    names=['shape', 'type', 'system', 'lam', 'beta', 'name',
                                           'r1', 'r2', 'phi1', 'phi2', 'note'],
                                    delim_whitespace=True)

        if not (all(dat['shape'] == 'SEGMENT') and all(dat['type'] == 'EX') and all(dat['r1'] == 0)):
            raise NotImplementedError('Only circular exclusion regions are supported up to now!')

        for r in dat.itertuples():
            ra, dec = r.lam, r.beta
            if r.system == 'GAL':
                ra, dec = galactic_to_radec(r.lam, r.beta)
            self.append(ExclusionRegion(r.name, ra, dec, r.r2))

    @property
    def names(self):
        return [r.name for r in self]

    def contains(self, test_ra, test_dec):
        mask = np.zeros_like(test_ra, dtype='bool')
        hit_regions = []
        for r in self:
            inside = r.contains(test_ra, test_dec)
            if inside.any():
                hit_regions.append(r)
            mask |= inside
        return mask, hit_regions

    def overlaps(self, test_ra, test_dec, radius):
        mask = np.zeros_like(test_ra, dtype='bool')
        hit_regions = []
        for r in self:
            overlap = r.overlaps(test_ra, test_dec, radius)
            if overlap.any():
                hit_regions.append(r)
            mask |= overlap
        return mask, hit_regions

    def get_region(self, name):
        for r in self:
            if r.name == name:
                return r
        raise ValueError('No region with name {} found!'.format(name))


class ExclusionRegion(object):
    def __init__(self, name, ra, dec, radius):
        self.name   = name
        self.ra     = ra
        self.dec    = dec
        self.radius = radius

    def contains(self, test_ra, test_dec):
        return angle_between(self.ra, self.dec, test_ra, test_dec) < self.radius

    def overlaps(self, test_ra, test_dec, test_radius):
        return angle_between(self.ra, self.dec, test_ra, test_dec) < self.radius + test_radius


def get_excluded_regions(ra, dec, radius):
    ers = ExclusionRegionSet()
    ers.read_from_file('/home/hfm/hess/hap-18-11/hdanalysis/lists/ExcludedRegions.dat')
    ers.read_from_file('/home/hfm/hess/hap-18-11/hdanalysis/lists/ExcludedRegions-stars.dat')
    return ers.overlaps(ra, dec, radius)[-1]
