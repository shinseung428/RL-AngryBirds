
import numpy as np
import tensorflow as tf
from architecture import *

class Agent():
	def __init__(self,config):


		self.screen_w, self.screen_h, self.channel = (config.screen_w, config.screen_h, config.channel_dim)
		self.batch_size = config.batch_size

		self.input_shape = [self.batch_size, self.screen_w, self.screen_h, self.channel]
		self.action_num = config.action_num
		
		self.learning_rate = config.learning_rate 

		self.build_model()
		self.build_loss()

		
	def update_model(self, batch_data):
		states, actions, rewards = batch_data


	def make_action(self, state):
	# possible action list:
	# 0 : mouseON and move + towards x axis
	# 1 : mouseON and move - towards x axis
	# 2 : mouseON and move + towards y axis
	# 3 : mouseON and move - towards y axis
	# 4 : mouse UP (resets xy position to the center of the slingshot)
	# 5 : do nothing
		pass

	def build_model(self):

		self.input = tf.placeholder(tf.float32, shape=self.input_shape)
		self.actions = self.Policy_Network(self.input)

		print "Created slingshot model ..."

	
	def build_loss(self):
		print "Created loss ..."


	def Policy_Network(self, input, name="policy_network"):

		with tf.variable_scope(name): 
			net = []
			conv1 = conv2d(input, self.channel, 512, 3, 2, name='conv1')
			#add batch norm
			net.append(conv1)

			conv2 = conv2d(conv1, 512, 256, 3, 2, name='conv2')
			#add batch norm
			net.append(conv2)

			conv3 = conv2d(conv2, 256, 128, 3, 2, name='conv3')
			#add batch norm
			net.append(conv3)

			conv4 = conv2d(conv3, 128, 64, 3, 2, name='conv4')
			#add batch norm
			net.append(conv4)

			#implement fc layer
			output = conv4

			return net, output
