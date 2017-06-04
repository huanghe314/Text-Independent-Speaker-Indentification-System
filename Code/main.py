"""
Real-Time Text-Independent Speaker Identification main system.

First training for 30 seconds for each speaker. 
Then testing for 60 seconds.

Author: He Huang and Shihong Fang 12/17/2015 
"""

import pyaudio
import struct
import math
import wave 
import mfcc
import Analysis
import numpy as np
import time
import training

# Initialization
n = 1
m = 1

Name = [ 0 for i in range(10)]

# Training Part
while m != 0:

    Name[n-1] = raw_input("What is your name? ")

    print "Hello, %s." % Name[n-1]

    BLOCKSIZE = 64  # Number of frames per block
    WIDTH = 2       # Number of bytes per sample
    CHANNELS = 1    # mono
    RATE = 44100    # Sampling rate (samples/second)
    RECORD_SECONDS = 30 # Time lasts for Recording 

    p1 = pyaudio.PyAudio()

    stream1 = p1.open(format = p1.get_format_from_width(WIDTH),
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    output = False)

    output_block = [0 for i in range(0, BLOCKSIZE)]

    output_wavefile = Name[n-1] + '.wav'
    output_wf = wave.open(output_wavefile, 'w')      # wave file
    output_wf.setframerate(RATE)
    output_wf.setsampwidth(WIDTH)
    output_wf.setnchannels(CHANNELS)

    num_blocks = int(RATE / BLOCKSIZE * RECORD_SECONDS)

    print('* Recording for {0:.3f} seconds '.format(RECORD_SECONDS))


# Start loop
    for i in range(0, num_blocks):

        input_string = stream1.read(BLOCKSIZE)   

        output_wf.writeframes(input_string)

    print('* Writing Done \n')

    stream1.stop_stream()
    stream1.close()
    p1.terminate()

    yn = raw_input("Do you want to change another recorder? [Y/N] ")

    if yn == 'Y':
        n = n + 1
    elif yn == 'N':
        m = 0

p_weight = [0 for i in range(n)]
Mean = [0 for i in range(n)]
Covar = [0 for i in range(n)]

for m in range(n):

	p_weight[m] = training.Training_feature_Weight(Name[m]+'.wav')

	Mean[m] = training.Training_feature_Mean(Name[m]+'.wav')

	Covar[m] = training.Training_feature_Covar(Name[m]+'.wav')

print '  Done  '

yn_2 = raw_input('Do you wanna test Now? [Y/N] ')


# Testing Part

if yn_2 == 'Y':

	BLOCKSIZE = 20000
	WIDTH = 2       # Number of bytes per sample
	CHANNELS = 1    # mono
	RATE = 44100    # Sampling rate (samples/second)

	list_buffer = np.zeros((45,13))
	M = 0
	
	def my_callback_fun(input_string, BLOCKSIZE, time_info, status):

		global list_buffer, M
		
		input_tuple = struct.unpack('h'*BLOCKSIZE, input_string)

		x = mfcc.mfcc_features(np.array(input_tuple))
		
		if M == 0:
			
			y = np.append(x,x,axis = 0)
		else:
			y = np.append(x,list_buffer,axis=0)
		
		# print list_buffer.shape

		# print x.shape

		# print y.shape

		Final = Analysis.GMM_identity(y, n, Name,p_weight,Mean,Covar)

		list_buffer = x 
		
		M = M + 1		

		print Name[Final] 

		return(None, pyaudio.paContinue)


	p = pyaudio.PyAudio()

	PA_format = p.get_format_from_width(WIDTH)

	stream = p.open(format = PA_format,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    output = False,
                    frames_per_buffer=BLOCKSIZE,
                    stream_callback = my_callback_fun) 
    
	stream.start_stream()

	time.sleep(60)

	stream.stop_stream()
	stream.close()
	p.terminate()
