
import numpy as np
import tensorflow as tf

class Agent():
	def __init__(self, 
				 screen_w, screen_h, channel
				 ):


		self.screen_w, self.screen_h, self.channel = (screen_w, screen_h, channel)
		self.batch_size = 64

		self.input_shape = [self.batch_size, self.screen_w, self.screen_h, self.channel]
		self.output_dim = 6
		
		self.learning_rate = 0.001 

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


	def Policy_Network(self, input):

		net = []

		return self.output_dim
