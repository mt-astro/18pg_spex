#! /Users/skywalker/anaconda3/bin/python

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.modeling.models import Gaussian1D
from astropy.stats import sigma_clip
import _pickle as pkl
from lmfit import Parameters, minimize
from define import *
from matplotlib.widgets import TextBox,CheckButtons,Slider
from scipy.optimize import curve_fit


LINES = {
	'Ha':6562.8,
	'Hb':4861.4,
	'Hg':4340.5,
	'Hd':4101.7,
	'HeI5876':5876.,
	'OIII':3760.,
	'NIII4100':4100., 
	'NIII4640':4640.,
	'HeII':4686., 
	'HeI6678':6678.,
	'NII5754':5754.,

}


HbetaFile='./LineStats_HbetaNIIIHeII.txt'


def main():

	fnames = glob.glob(PKL_DIR+'/*.pkl')
	spex = [open_pkl(ff) for ff in fnames]
	mjds = [spec.mjd for spec in spex]
	order = np.array(mjds).argsort()

	for ii in order:
		spec = spex[ii]
		print(spec.mjd, spec.date)
		#fitOIIIRegion(spec.wl, spec.fl)		
		fitHbetaRegion(spec.wl, spec.fl)
		fitNIIRegion(spec.wl, spec.fl)
		fitHalphaRegion(spec.wl, spec.fl, spec.err)



def fitHalphaRegion(wl, fl, err):
	
	wl_width = vel2wl(6562.8, 40000.)
	wl_low, wl_high = 6562.8 - wl_width, 6562.8 + wl_width
	keep = np.where( (wl >= wl_low) & (wl <= wl_high) )[0]

	plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
	plt.title('Click 4 points to identify the continuum fitting regions')
	continuum_points = plt.ginput(4, 30)
	plt.close()
	continuum = fit_continuum(wl[keep], fl[keep], continuum_points)
	resid_fl = fl[keep] - continuum

	params = HalphaPlotter(wl[keep], resid_fl)

def fitHbetaRegion(wl, fl):
	wl_width = vel2wl(4700., 40000.)
	keep = np.where(
		(wl >= 4700.-wl_width) &
		(wl <= 4700.+wl_width)
		)[0]

	plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
	plt.title('Click 4 points to identify the continuum fitting regions')
	continuum_points = plt.ginput(4, 30)
	plt.close()
	continuum = fit_continuum(wl[keep], fl[keep], continuum_points)
	resid_fl = fl[keep] - continuum



	params = HbetaPlotter(wl[keep], resid_fl)
	line_data = {
		'Hbeta':{'FWHM':params.HbFWHM, 'Amp':params.Hb_amp},
		'NIII':{'FWHM':params.NIIIFWHM, 'Amp':params.NIII_amp},
		'HeII':{'FWHM':params.HeIIFWHM, 'Amp':params.HeII_amp}
	}
	return line_data


def fitOIIIRegion(wl, fl):
	if wl.min() > 4000.: 
		print('No data near Hdelta/OIII, press Enter to continue')
		input()
		return
	
	for name, line in zip(['Hd', 'Hg', 'OIII'], [4102., 4340.5, 3760.0]):
		continue
		keep = np.where((wl >= line-300.)&(wl <= line+300.))[0]
		plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
		plt.axvline(line, 0, 1, color='g', ls='--')
		plt.title('Click 3 points: left edge, peak, right edge')
		points = plt.ginput(3, 20)
		plt.close()
		if len(points) < 3: 
			print('Skipping line fit...')
			continue
		fitGauss(wl[keep], fl[keep], points=points)




	params = OIIIPlotter(wl[keep], resid_fl)
	#line_data = {
	#	'Hbeta':{'FWHM':params.HbFWHM, 'Amp':params.Hb_amp},
	#	'NIII':{'FWHM':params.NIIIFWHM, 'Amp':params.NIII_amp},
	#	'HeII':{'FWHM':params.HeIIFWHM, 'Amp':params.HeII_amp}
	#}
	#return line_data




class HalphaPlotter:
	def __init__(self, wl, fl):
		self.wl = wl
		self.fl = fl

		self.fig, self.ax = plt.subplots(figsize=(10,8))
		self.ax.plot(self.wl, self.fl, 'k-', drawstyle='steps-mid')

		self.gauss_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'r:')
		self.HaN_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'b:')
		self.comp_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'm--')

		#self.ax.set_xlim(self.xlims)
		plt.subplots_adjust(bottom=0.22, top=0.95, left=0.1, right=0.95)

		# set up sliders for params
		self.HaB_vel = 5000.
		axbox = plt.axes([0.1, 0.05, 0.3, 0.05])
		Ha_vel = Slider(axbox, 'Halpha Vel\n [km/s]', 1000., 20000, valinit=self.HaB_vel, valstep=1000., valfmt='%.1f')
		Ha_vel.on_changed(self._updateHaBVel)

		self.HaB_amp = np.interp(6563., self.wl, self.fl)
		axbox = plt.axes([0.1, 0.0, 0.3, 0.05])
		Ha_amp = Slider(axbox, 'HaBroad\nAmp.', 0.0, 1.5*self.fl.max(), valinit=self.HaB_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		Ha_amp.on_changed(self._updateHaBAmp)

		self.HaN_amp = 0.0
		axbox = plt.axes([0.5, 0.0, 0.3, 0.05])
		HaN_amp = Slider(axbox, 'HaNarrow\nAmp', 0.0, self.fl.max(), valinit=self.HaN_amp, valstep=self.fl.max()/100., valfmt='%.1e')
		HaN_amp.on_changed(self._updateHaNAmp)

		self.HaN_v3l = 3000.
		axbox = plt.axes([0.5, 0.05, 0.3, 0.05])
		HaN_vel = Slider(axbox, 'HaNarrow\nFWHM', 1000.0, 10000, valinit=self.HaN_vel, valstep=250.0, valfmt='%.1f')
		HaN_vel.on_changed(self._updateHaNFWHM)

		
		self.Ha_mean = 6563.0
		self.Ha_shift = 0.0
		axbox = plt.axes([0.1, 0.1, 0.8, 0.05])
		Ha_mean = Slider(axbox, 'Ha Shift', -3000.0, 3000.0, valinit=self.Ha_shift, valstep=100.)
		Ha_mean.on_changed(self._updateHaShift)

		plt.show()


	def _updateHaVel(self, text):
		self.Ha_vel = float(text)
		self.update()

	def _updateHaAmp(self, text):
		self.Ha_amp = float(text)
		self.update()

	def _updateHaShift(self, text):
		self.Ha_shift = float(text)
		self.update()

	def update(self):
		HaB_gauss = Gaussian1D(amplitude=self.HaB_amp, stddev=vel2wl(6562.8, self.HaB_vel)/2.35, mean=self.Ha_mean + vel2wl(self.Ha_mean, self.Ha_shift))
		HaB_fl = HaB_gauss(self.wl)
		self.HaB_spec.set_ydata(gauss_fl)

		HaN_gauss = Gaussian1D(amplitude=self.HaN_gauss, sttdev=vel2wl(self.Ha_mean, self.HaN_vel)/2.35, mean=self.Ha_mean + vel2wl(self.Ha_mean, self.Ha_shift))
		HaN_fl = HaN_gauss(self.wl)
		self.HaN_spec.set_ydata(HaN_fl)

		comp = HaN_fl + HaB_fl
		self.comp_spec.set_ydata(comp)

		self.fig.canvas.draw()


class HbetaPlotter:
	def __init__(self, wl, fl):
		self.wl = wl
		self.fl = fl

		self.fig, self.ax = plt.subplots(figsize=(10,8))
		self.ax.plot(self.wl, self.fl, 'k-', drawstyle='steps-mid')

		self.Hb_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'r:', label='Hbeta')
		self.NIII_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'b:', label='NIII')
		self.HeII_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'y:', label='HeII')
		self.comp_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'm--', label='Composite')

		#self.ax.set_xlim(self.xlims)
		plt.subplots_adjust(bottom=0.3, top=0.95, left=0.1, right=0.95)

		# set up sliders for params


		
		self.Hb_mean = 4861.
		self.NIII_mean = 4640.
		self.HeII_mean = 4686.

		# set up sliders for params
		self.NIII_amp = np.interp(4640., self.wl, self.fl)/2.0
		axbox = plt.axes([0.1, 0.15, 0.3, 0.05])
		NIII_amp = Slider(axbox, 'NIII Amp.\n[erg/s/cm2/A]', 0.0, 1.5*self.fl.max(), valinit=self.NIII_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		NIII_amp.on_changed(self._updateNIIIAmp)

		self.Hb_amp = np.interp(4861., self.wl, self.fl)
		axbox = plt.axes([0.1, 0.05, 0.3, 0.05])
		Hb_amp = Slider(axbox, 'Hb Amp.\n[erg/s/cm2/A]', 0.0, 1.5*self.fl.max(), valinit=self.Hb_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		Hb_amp.on_changed(self._updateHbAmp)

		self.HeII_amp = np.interp(self.HeII_mean, self.wl, self.fl)/2.0
		axbox = plt.axes([0.1, 0.1, 0.3, 0.05])
		HeII_amp = Slider(axbox, 'HeII Amp.\n[erg/s/cm2/A]', 0.0, 1.5*self.fl.max(), valinit=self.HeII_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		HeII_amp.on_changed(self._updateHeIIAmp)

		self.HeIIFWHM = 10000.
		axbox = plt.axes([0.6, 0.1, 0.3, 0.05])
		fwhm = Slider(axbox, 'HeII\nFWHM', 1000., 20000., valinit=self.HeIIFWHM, valstep=1000., valfmt="%.1f")
		fwhm.on_changed(self._updateHeIIFWHM)

		self.HbFWHM = 10000.
		axbox = plt.axes([0.6, 0.05, 0.3, 0.05])
		HbFWHM = Slider(axbox, 'Hb\nFWHM', 1000.0, 20000, valinit=self.HbFWHM, valstep=1000., valfmt="%.1f")
		HbFWHM.on_changed(self._updateHbFWHM)

		self.NIIIFWHM = 10000.0
		axbox = plt.axes([0.6, 0.15, 0.3, 0.05])
		NIIIFWHM = Slider(axbox, 'NIII\nFWHM', 1000., 20000., valinit=self.NIIIFWHM, valstep=1000., valfmt='%.1f')
		NIIIFWHM.on_changed(self._updateNIIIFWHM)
	
		self.changed = False
		plt.legend()
		plt.show()

	def _updateNIIIFWHM(self, text):
		self.NIIIFWHM = float(text)
		self.update()

	def _updateHbAmp(self, text):
		self.Hb_amp = float(text)
		self.update()

	def _updateHbFWHM(self, text):
		self.HbFWHM = float(text)
		self.update()

	def _updateNIIIAmp(self, text):
		self.NIII_amp = float(text)
		self.update()

	def _updateHeIIAmp(self, text):
		self.HeII_amp = float(text)
		self.update()

	def _updateHeIIFWHM(self, text):
		self.HeIIFWHM = float(text)
		self.update()

	def update(self):
		self.changed = True
		Hb_gauss = Gaussian1D(amplitude=self.Hb_amp, stddev=vel2wl(4861, self.HbFWHM)/2.35, mean=self.Hb_mean)
		Hb_fl = Hb_gauss(self.wl)
		self.Hb_spec.set_ydata(Hb_fl)

		NIII_gauss = Gaussian1D(amplitude=self.NIII_amp, stddev=vel2wl(4640., self.NIIIFWHM)/2.35, mean=self.NIII_mean)
		NIII_fl = NIII_gauss(self.wl)
		self.NIII_spec.set_ydata(NIII_fl)

		HeII_gauss = Gaussian1D(amplitude=self.HeII_amp, stddev=vel2wl(self.HeII_mean, self.HeIIFWHM)/2.35, mean=self.HeII_mean)
		HeII_fl = HeII_gauss(self.wl)
		self.HeII_spec.set_ydata(HeII_fl)

		comp_fl = Hb_fl + NIII_fl + HeII_fl
		self.comp_spec.set_ydata(comp_fl)

		self.fig.canvas.draw()


class OIIIPlotter:
	def __init__(self, wl, fl):
		self.wl = wl
		self.fl = fl

		self.fig, self.ax = plt.subplots(figsize=(10,8))
		self.ax.plot(self.wl, self.fl, 'k-', drawstyle='steps-mid')

		self.OIII_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'r:', label='OIII3760A')
		self.Hd_spec, = self.ax.plot(self.wl, np.zeros_like(self.wl), 'b:', label='Hdelta')
		plt.legend()
		#self.ax.set_xlim(self.xlims)
		plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)

		# set up sliders for params


		
		self.OIII_mean = 3760.
		self.Hd_mean = 4102.

		# set up sliders for params
		self.OIII_amp = np.interp(3760., self.wl, self.fl)
		axbox = plt.axes([0.1, 0.05, 0.3, 0.05])
		OIII_amp = Slider(axbox, 'OIII Amp.\n[erg/s/cm2/A]', 0.0, 1.5*self.fl.max(), valinit=self.OIII_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		OIII_amp.on_changed(self._updateOIIIAmp)

		self.OIIIFWHM = 10000.0
		axbox = plt.axes([0.6, 0.05, 0.3, 0.05])
		OIIIFWHM = Slider(axbox, 'OIII FWHM', 1000., 20000., valinit=self.OIIIFWHM, valstep=1000., valfmt='%.1f')
		OIIIFWHM.on_changed(self._updateOIIIFWHM)

		# set up sliders for params
		self.Hd_amp = np.interp(4102., self.wl, self.fl)
		axbox = plt.axes([0.1, 0.1, 0.3, 0.05])
		Hd_amp = Slider(axbox, 'HdAmp', 0.0, 1.5*self.fl.max(), valinit=self.Hd_amp, valstep=(self.fl.max()-self.fl.mean())/100.0, valfmt="%.1e")
		Hd_amp.on_changed(self._updateHdAmp)

		self.HdFWHM = 10000.0
		axbox = plt.axes([0.6, 0.1, 0.3, 0.05])
		HdFWHM = Slider(axbox, 'HdFWHM', 1000., 20000., valinit=self.HdFWHM, valstep=1000., valfmt='%.1f')
		HdFWHM.on_changed(self._updateHdFWHM)

		self.update()
	
		self.changed = False
		plt.show()

	def _updateOIIIFWHM(self, text):
		self.OIIIFWHM = float(text)
		self.update()

	def _updateOIIIAmp(self, text):
		self.OIII_amp = float(text)
		self.update()

	def _updateHdFWHM(self, text):
		self.HdFWHM = float(text)
		self.update()
	def _updateHdAmp(self, text):
		self.Hd_amp = float(text)
		self.update()

	def update(self):
		self.changed = True

		OIII_gauss = Gaussian1D(amplitude=self.OIII_amp, stddev=vel2wl(self.OIII_mean, self.OIIIFWHM)/2.35, mean=self.OIII_mean)
		OIII_fl = OIII_gauss(self.wl)
		self.OIII_spec.set_ydata(OIII_fl)

		Hd_gauss = Gaussian1D(amplitude=self.Hd_amp, stddev=vel2wl(self.Hd_mean, self.HdFWHM)/2.35, mean=self.Hd_mean)
		Hd_fl = Hd_gauss(self.wl)
		self.Hd_spec.set_ydata(Hd_fl)

		self.fig.canvas.draw()




def fitGauss(wl, fl, points):
	left = points[0][0]
	right = points[-1][0]
	center = points[1][0]
	amp = points[1][1] - np.median(fl)


	slope = (points[-1][1] - points[0][1]) / (right - left)
	yint = points[0][1] / (slope * points[0][0])

	keep = np.where( (wl >= left) & (wl <= right) )[0]

	print(left, center, right, amp, slope, yint)
	p0 = [center, 9000.0, np.interp(center, wl, fl), slope, yint]
	bounds = (
		[center-10., 1000.0, 0.0, 1e-19, 0.4],
		[center+10., 20000.0, 2.0*fl[keep].max(), 1e-17, 0.6]
		)
	pfit, pcov = curve_fit(simple_gauss, wl[keep], fl[keep], p0=p0, bounds=bounds)
	perr = np.sqrt(np.diag(pcov))
	plt.plot(wl[keep], fl[keep], 'k-', drawstyle='steps-mid')
	plt.plot(wl, simple_gauss(wl, *pfit), 'r--')
	plt.show()

def simple_gauss(wl, mean, vel, amp, slope, yint):
	print(slope, yint)
	continuum = wl * slope + yint
	gauss = Gaussian1D(amplitude=amp, stddev=vel2wl(mean, vel)/2.35, mean=mean)
	return gauss(wl) + continuum

def minimizer_fcn(params, wl, fl, mask=None):
	print(wl.size, fl.size)
	model = redLineModel(params, wl)
	resid = (model - fl)**2.0
	if mask is None:
		mask = sigma_clip(resid, maxiters=1).mask
	resid[mask] = 0.0
	print(wl.size, fl.size, resid.size)

	return resid

def redLineModel(params, wl):
	values = params.valuesdict()

	Ha_gauss = Gaussian1D(amplitude=values['amp_Ha'], mean=values['wl_Ha'], stddev=values['width_Ha'])
#	HeI_gauss = Gaussian1D(amplitude=values['amp_HeI'], mean=values['wl_HeI'], stddev=values['width_HeI'])
	HeI_gauss = Gaussian1D(amplitude=0)
	continuum = values['slope']*wl + values['yint']


	model = Ha_gauss(wl) + HeI_gauss(wl) + continuum
	return model

def fit_continuum(wl, fl, points, order=1):
	(x0, x1, x2, x3) = [val[0] for val in points]
	cr = np.where(
		( (wl >= x0) & (wl <= x1) ) |
		( (wl >= x2) & (wl <= x3) )
		)[0]
	continuum = np.poly1d(np.polyfit(wl[cr], fl[cr], order))
	return continuum(wl)



def vel2wl(cen_wl, vel):
	# vel in km/s
	# cen_wl in A
	return vel / 299792.458 * cen_wl



def open_pkl(fname):
	with open(fname, 'rb') as p: return pkl.load(p)





if __name__=='__main__':
	main()