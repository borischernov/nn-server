'''
Network class contains all the NN-related functions
'''

import caffe
import numpy as np
import threading

class Network:
	def __init__(self, config, timeout = 10):
		"""
		Initialize class instance with configuration object "config"
		"""
		self.net = caffe.Net(config['model'], caffe.TEST, weights = config['weights'])
		
		# Configure input preprocessing
		incfg = config['input']
		
		self.in_blob = incfg['blob']
		self.transformer = caffe.io.Transformer({'data': self.net.blobs[self.in_blob].data.shape})
		
		if 'mean' in incfg:
			self.transformer.set_mean('data', np.array(incfg['mean']))
		if 'mean_file' in incfg:
			self.transformer.set_mean('data', np.load(incfg['mean_file']).mean(1).mean(1))
		
		if 'transpose' in incfg:
			self.transformer.set_transpose('data', tuple(incfg['transpose']))
		if 'channel_swap' in incfg:
			self.transformer.set_channel_swap('data', tuple(incfg['channel_swap']))
		if 'raw_scale' in incfg:
			self.transformer.set_raw_scale('data', incfg['raw_scale'])

		# Set number of samples in input batch to 1
		in_shape = tuple(self.net.blobs[self.in_blob].shape)
		self.net.blobs[self.in_blob].reshape(*((1,) + in_shape[1:]))

		# Configure output
		outcfg = config['output']
		self.out_blob = outcfg['blob']
		
		self.out_normalize = False
		if 'normalize' in outcfg:
			self.out_normalize = outcfg['normalize']
			
		self.lock = threading.Lock()
		self.timeout = timeout
		
	def forward(self, im):
		"""
		Process image im
		"""
		if not self.lock.acquire(blocking = True, timeout = self.timeout):
			raise TimeoutError("Failed to acquire lock on the network")
		
		try:
			# Preprocess input
			self.net.blobs[self.in_blob].data[...] = self.transformer.preprocess('data', im)
			
			# Forward Pass
			self.net.forward()
			
			# Process output
			out = self.net.blobs[self.out_blob].data.reshape(-1)
			
			if self.out_normalize:
				norm = np.linalg.norm(out)
				if norm != 0.:
					out /= norm
		finally:
			self.lock.release()

		return out
