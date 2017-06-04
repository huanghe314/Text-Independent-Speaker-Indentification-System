""" This file contains a GMM class, which also calculates the means, weights and covariances of each
Gaussian model. In addition, the file return those values for further study.

Author: He Huang and Shihong Fang 12/17/2015 
"""
import numpy as np 
from sklearn import cluster, mixture 
import scipy.io.wavfile as wav
import mfcc 

class GMM:

	def __init__(self, M, input_audio_file):

		self.M = M 

		(rate,sig) = wav.read(input_audio_file)            # get audio data and sampling rate 

		feature_vectors = mfcc.mfcc_features(sig)      # get feature marix of audio data 

		self.features = feature_vectors

	def GMM_Model_Mean(self):

		mean = mixture.GMM(n_components = self.M, min_covar = 0.01, n_init = 10).fit(self.features).means_

		return mean 

	def GMM_Model_Weight(self):

		weight = mixture.GMM(n_components = self.M, min_covar = 0.01, n_init = 10).fit(self.features).weights_

		return weight 

	def GMM_Model_Covar(self):

		covar = mixture.GMM(n_components = self.M, min_covar = 0.01, n_init = 10).fit(self.features).covars_

		return covar 


