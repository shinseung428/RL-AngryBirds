
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
		self.epsilon = config.epsilon

		self.build_model()
		self.build_loss()

		
	def update_model(self, batch_data):
		# states, actions, rewards = zip(*batch_data)
		pass




	def get_action(self, state, x_mouse, y_mouse, mouse_pressed):
		# possible action list:
		# 0 : mouseON and move + towards x axis
		# 1 : mouseON and move - towards x axis
		# 2 : mouseON and move + towards y axis
		# 3 : mouseON and move - towards y axis
		# 4 : mouse UP (resets xy position to the center of the slingshot)
		# 5 : do nothing
			    #Selecting actions without delay 
		#select random action
		xy_distance = 5
		if self.epsilon > np.random.uniform(0,1):
			action = 5
		else:
			action = np.random.randint(6, size=(1))[0]


		if action == 0:
			mouse_pressed = True
			x_mouse += xy_distance
		elif action == 1:
			mouse_pressed = True
			x_mouse -= xy_distance
		elif action == 2:
			mouse_pressed = True
			y_mouse += xy_distance
		elif action == 3:
			mouse_pressed = True
			y_mouse -= xy_distance
		elif action == 4:
			mouse_pressed = False
			#reset the position of the mouse to the center point of the slingshot
			x_mouse, y_mouse = (130, 426)
		elif action == 5:#do nothing
			pass

		#bound the movement of the mouse
		if x_mouse < 100:
			x_mouse = 100
		if x_mouse > 250:
			x_mouse = 250
		if y_mouse < 370:
			y_mouse = 370
		if y_mouse > 550:
			y_mouse = 550

		return action, x_mouse, y_mouse, mouse_pressed


	def build_model(self):

		self.input = tf.placeholder(tf.float32, shape=self.input_shape)
		self.actions = self.Policy_Network(self.input)

		print "Created slingshot model ..."

	
	def build_loss(self):
		print "Created loss ..."


	def Policy_Network(self, input, name="policy_network"):

		with tf.variable_scope(name): 
			net = []
			# conv1 = conv2d(input, self.channel, 512, 3, 2, name='conv1')
			# conv1 = batchnorm(conv1, name='bn1')
			# conv1 = tf.nn.relu(conv1)
			# net.append(conv1)
			conv1 = tf.contrib.layers.conv2d(input, 512, 3, stride=2, scope='conv1')
			conv1 = tf.contrib.layers.batch_norm(conv1, scope='bn1')
			conv1 = tf.nn.relu(conv1)
			net.append(conv1)

			conv2 = tf.contrib.layers.conv2d(conv1, 256, 3, stride=2, scope='conv2')
			conv2 = tf.contrib.layers.batch_norm(conv2, scope='bn2')
			conv2 = tf.nn.relu(conv2)
			net.append(conv2)

			conv3 = tf.contrib.layers.conv2d(conv2, 128, 3, stride=2, scope='conv3')
			conv3 = tf.contrib.layers.batch_norm(conv3, scope='bn3')
			conv3 = tf.nn.relu(conv3)
			net.append(conv3)

			conv4 = tf.contrib.layers.conv2d(conv3, 64, 3, stride=2, scope='conv4')
			conv4 = tf.contrib.layers.batch_norm(conv4, scope='bn4')
			conv4 = tf.nn.relu(conv4)
			net.append(conv4)

			flattned = tf.contrib.layers.flattned(conv4, scope='flattned')
			fc1 = tf.contrib.layers.fully_connected(flattned, 128, scope='fc1')
			fc2 = tf.contrib.layers.fully_connected(fc1, self.action_num, scope='fc2')
			#implement fc layer
			output = fc2
			
			return net, output
