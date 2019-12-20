import numpy as np
from astropy import units as u
from filters import Swope
from scipy.interpolate import interp1d
import os, sys, glob

import matplotlib.pyplot as plt

from define import *

NEWSPEC_DIR = './newspec_dec19'
CALSPEC_DIR='./newspec_cal'


COLORS = {
	'u':'c',
	'B':'b',
	'V':'g',
	'g':'lightgreen',
	'r':'orange',
	'i':'m'
}


def main():
	fnames = glob.glob(NEWSPEC_DIR+'/*.flm')
	spex = [Spectrum(ff) for ff in fnames]

	global JDs
	spec_names, JDs = np.genfromtxt(NEWSPEC_DIR+'/spec_info.txt', unpack=True, dtype=str, skip_header=1)
	spec_names = spec_names.tolist()
	JDs = JDs.astype(float) - 2400000.5
	phot_interp = readPhot()

	for spec in spex:
		baseff = os.path.basename(spec.fname)
		baseff2 = np.str.replace(baseff, '.flm', '.txt')
		idx = spec_names.index(baseff2)
		mjd = JDs[idx]
		
		scales = []
		effwls = []
		for band, intpr in phot_interp.items():
			#if band in ['u', 'i']: continue
			mag = intpr(mjd)
			filt_obj = eval('Swope.%s' % band)
			trans = filt_obj(spec.wl)
			scale = SynPhot(spec.wl, spec.fl, mag, trans)

			scales.append(scale)
			effwls.append(filt_obj.wl_eff)

		#scale_intpr = interp1d(effwls, scales, kind='slinear', bounds_error=False, fill_value=(min(scales), max(scales)))
		scale_intpr = np.poly1d(np.polyfit(effwls, scales, 2))
		spec.fl *= scale_intpr(spec.wl)
		with open(CALSPEC_DIR+'/'+baseff, 'w') as ofile:
			for w_f in zip(spec.wl/(1.0+REDSHIFT), spec.fl): ofile.write('%lf %le\n' % w_f)
		

def readPhot():
	fname = NEWSPEC_DIR+'/asassn-18pg_phot_comb_smallap.txt'
	phot= {}
	for line in open(fname, 'r').readlines():
		mjd, band, mag, err, _ = line.strip().split()
		if band not in phot.keys():
			phot[band] = {'mjd':[], 'mag':[], 'err':[]}
		phot[band]['mjd'].append( float(mjd) )
		phot[band]['mag'].append( float(mag) )
		phot[band]['err'].append( float(err) )

	interpers = {}
	for band, data in phot.items():
		intpr = interp1d(data['mjd'], data['mag'], bounds_error=False, fill_value='extrapolate', kind='slinear')
		interpers[band] = intpr

	return interpers


def SynPhot(wl, fl, mag, resp):
	resp[~np.isfinite(resp)] = 0.0
	nu = (wl*u.AA).to(u.Hz, equivalencies=u.spectral())
	fl_Hz = fl*(u.erg/u.s/u.cm**2.0/u.AA).to(u.erg/u.s/u.cm**2.0/u.Hz, equivalencies=u.spectral_density(wl*u.AA))
	scale_fl = fl_Hz*resp
	synmag = -2.5*np.log10( np.trapz(scale_fl) / np.trapz(resp)) - 48.6
	diff = mag - synmag
	scale = 10.0**(-0.4*diff)
	return scale


if __name__=='__main__':
	main()