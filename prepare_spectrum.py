import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import os, sys, glob
from astropy.io import fits
from astropy.time import Time
from scipy.interpolate import interp1d
import argparse

from define import *

def main(fname, plot=False):
	spec = Spectrum(fname)
	spec.mjd = date2mjd(spec.date)
	spec.err = estimate_uncert(spec.wl, spec.fl)
	spec.mask = make_mask(spec.wl, spec.fl)


	if plot:
		print('Plotting spectrum...')
		plt.plot(spec.wl, spec.fl, 'k-', drawstyle='steps-mid')
		plt.xlabel('Wavelength [A]')
		plt.ylabel('Flux [ergs/s/cm2/A]')
		plt.show()
	
	with open(spec.pklfile, 'wb') as p:
		pickle.dump(spec, p)

def findcontinuumRegions(wl, fl, stepsz=500.):
	start = wl.min()
	end = wl.max()

	points = []
	while start < end:
		keep = np.where( (wl >= start) & (wl <= start+stepsz))[0]
		plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
		plt.xlabel('Wavelength [A]')
		plt.ylabel('Flux')
		plt.title('"m" to mark, "Enter" to exit, "backspace" to delete')
		p = plt.ginput(-1, 30, True, mouse_add=3, mouse_pop=None, mouse_stop=None)
		plt.close()
		points += p

		start += stepsz

	if len(points) % 2 != 0:
		raise ValueError('Uneven number of points selected for continuum fitting!')


	regs= []
	while len(points) > 0:
		p0 = points.pop(0)
		p1 = points.pop(0)
		reg = (p0[0], p1[0])
		regs.append(reg)

	return regs

def est_rms(wl, fl, locs):
	keep = np.where( (wl >= locs[0]) & (wl <= locs[1]))[0]
	linfit = np.poly1d(np.polyfit(wl[keep], fl[keep], 1))
	resid = fl[keep] - linfit(wl[keep])
	rms = resid.std(ddof=1)
	return rms

def estimate_uncert(wl, fl):
	continua_reg = findcontinuumRegions(wl, fl)
	center_waves = [np.mean(a) for a in continua_reg]
	err_est = [est_rms(wl, fl, cr) for cr in continua_reg]
	interper = interp1d(center_waves, err_est, kind='cubic', bounds_error=False, fill_value='extrapolate')
	err = interper(wl)
	return err

def make_mask(wl, fl):
	masked_reg = findMaskRegions(wl, fl)
	mask_idx = [1 if i in masked_reg else 0 for i in range(wl.size)]
	mask = np.array(mask_idx)
	return mask

def findMaskRegions(wl, fl, stepsz=300.):
	start = wl.min()
	end = wl.max()

	points = []
	while start < end:
		keep = np.where( (wl >= start-50.) & (wl <= start+stepsz+50.))

		plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
		plt.xlabel('Wavelength [A]')
		plt.ylabel('Flux')
		plt.title('"m" to mark, "Enter" to exit, "backspace" to delete')
		p = plt.ginput(-1, 30, True, mouse_add=3, mouse_pop=None, mouse_stop=None)
		plt.close()
		points += p

		start += stepsz


	if len(points) % 2 != 0:
		raise ValueError('Uneven number of points selected for continuum fitting!')

	regs= []
	while len(points) > 0:
		p0 = points.pop(0)
		p1 = points.pop(0)
		reg = (p0[0], p1[0])
		regs.append(reg)

	mask_idx = []
	for (x0, x1) in regs:
		keep = np.where((wl >= x0) & (wl <= x1))[0].tolist()
		mask_idx += keep

	mask = np.array( [ 1 if i in mask_idx else 0 for i in range(wl.size)])
	return mask



	return regs



def date2mjd(date):
	if len(date) != 6: raise ValueError('Unexpectded date length: %d, expected 6' % len(date))
	longdate = '20'+date[:2]+'-'+date[2:4] + '-'+date[4:]
	return Time(longdate, format='iso').mjd


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Prepare a txt spectrum for processing')
	parser.add_argument('fname', help='Spectrum file name to process.', type=str)
	parser.add_argument('--plot', '-p', help='Show plots? Default: False', default=False, action='store_true')

	args = parser.parse_args()
	main(args.fname, args.plot)