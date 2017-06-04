""" This file contains algorithm we used for testing. Basically, here we use log-liklyhood function
to decide which model matches the sound sample best.

Author: He Huang and Shihong Fang 12/17/2015 

"""
from scipy.stats import multivariate_normal as mu_nor
import numpy as np  
import training 
import math

def GMM_identity(x, n, Name,p_weight,Mean,Covar):

	log_sum = [0 for i in range(n)]

	for m in range(n):

		# p_weight = training.Training_feature_Weight(Name[m]+'.wav')

		# Mean = training.Training_feature_Mean(Name[m]+'.wav')

		# Covar = training.Training_feature_Covar(Name[m]+'.wav')

		for i in range(x.shape[0]):

			sum_pb = 0

			for j in range(32):

				p = p_weight[m][j]*mu_nor.pdf(x[i],mean = Mean[m][j],cov = Covar[m][j])

				sum_pb = p + sum_pb

				# print sum_pb

			if sum_pb > 0:

				log_sum[m] = math.log(sum_pb) + log_sum[m]

			else:

				log_sum[m] = -9999999999999999+log_sum[m]



			# print log_sum

	return log_sum.index(max(log_sum))