import os
import numpy as np

REDSHIFT = 0.018
SPEC_DIR = './18pg_cal_spec'
PKL_DIR = './pkl_files'

class Spectrum:
	def __init__(self, fname):
		self.fname = fname
		self.wl, self.fl = np.genfromtxt(self.fname, unpack=True, dtype=float, comments='#')
		baseff = os.path.basename(self.fname)
		try:
			_, self.instr, self.date, _ = baseff.split('_', 3)
		except:
			_, self.instr, self.date = baseff[:-4].split('_')

		self.pklfile = PKL_DIR+'/%s-%s.pkl' % (self.instr, self.date)
		if os.path.exists(self.pklfile):
			raise IOError('spectrum pkl file %s already exists, delete to remake' % self.pklfile)

