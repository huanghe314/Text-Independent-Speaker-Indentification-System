"""
Extract MFCC features and perform Mean Normalizaion

Author: He Huang and Shihong Fang 12/17/2015 
"""
import base
import numpy as np 

def mfcc_features(sig):

	mfcc_features = base.mfcc(sig, samplerate = 44100, winlen=0.02, winstep = 0.01, numcep = 13, nfilt = 40)

	# Mean Normalizaion for feature vectors.

	mean_vector = np.mean(mfcc_features, axis = 0)

	mfcc_features_update = mfcc_features - mean_vector

	return mfcc_features_update

