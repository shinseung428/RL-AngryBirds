
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
		self.momentum = config.momentum
		self.epsilon = config.epsilon

		self.logpoint = config.logpath
		self.modelpath = config.modelpath


		self.build_model()
		self.build_loss()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def update_model(self, states, actions, advantages):
		#normalize rewards
		advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

		batch_feed = {self.input_state: states,
					  self.input_act: actions,
					  self.input_adv: advantages}
		loss, _ = self.sess.run([self.loss, self.train], feed_dict=batch_feed)

		self.loss = tf.Print(self.loss, [self.loss], )
		return loss



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
			# self.sess.run(self.actions, feed_dict={self.input_state: state})
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

		self.input_state = tf.placeholder(tf.float32, shape=self.input_shape)
		self.net, self.actions = self.Policy_Network(self.input_state)

		self.input_act = tf.placeholder(tf.int32)
		self.input_adv = tf.placeholder(tf.float32)

		self.trainable_vars = tf.trainable_variables()

		print "Created slingshot model ..."

	
	def build_loss(self):
		self.log_prob = tf.log(tf.nn.softmax(self.actions))

		# get log probs of actions from episode
		indices = tf.range(0, tf.shape(self.log_prob)[0]) * tf.shape(self.log_prob)[1] + self.input_act
		act_prob = tf.gather(tf.reshape(self.log_prob, [-1]), indices)

		# surrogate loss
		self.loss = -tf.reduce_sum(tf.multiply(act_prob, self.input_adv))		

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='AdamOpt')
		self.train = self.optimizer.minimize(self.loss, var_list=self.trainable_vars)


		print "Created loss ..."


	def Policy_Network(self, input, name="policy_network"):

		with tf.variable_scope(name): 
			net = []
			# conv1 = conv2d(input, self.channel, 512, 3, 2, name='conv1')
			# conv1 = batchnorm(conv1, name='bn1')
			# conv1 = tf.nn.relu(conv1)
			# net.append(conv1)
			conv1 = tf.contrib.layers.conv2d(input, 512, 5, stride=3, scope='conv1')
			conv1 = tf.contrib.layers.batch_norm(conv1, scope='bn1')
			conv1 = tf.nn.relu(conv1)
			net.append(conv1)

			conv2 = tf.contrib.layers.conv2d(conv1, 256, 5, stride=2, scope='conv2')
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

			flattened = tf.contrib.layers.flatten(conv4, scope='flattened')
			fc1 = tf.contrib.layers.fully_connected(flattened, 200, scope='fc1')
			fc2 = tf.contrib.layers.fully_connected(fc1, self.action_num, scope='fc2')
			#implement fc layer
			output = fc2

			return net, output
