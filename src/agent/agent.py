
import numpy as np
import tensorflow as tf
from architecture import *

class Agent():
	def __init__(self,config):

		self.gamma = .99               # discount factor for reward
		self.decay = 0.99    


		self.screen_w, self.screen_h, self.channel = (config.screen_w, config.screen_h, config.channel_dim)
		self.batch_size = config.batch_size

		self.input_shape = [None, self.screen_w, self.screen_h, self.channel]
		self.action_num = config.action_num
		
		self.learning_rate = config.learning_rate 
		self.momentum = config.momentum

		self.graphpath = config.graphpath
		self.modelpath = config.modelpath


		self.build_model()
		self.build_loss()

		self.summary = tf.summary.merge([self.loss_graph])

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

		self.writer = tf.summary.FileWriter(self.graphpath, self.sess.graph)

	def update_model(self, states, actions, advantages, counter):
		#normalize input
		states = np.asarray(states, dtype=np.float32)/255.0
		#normalize rewards
		advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

		batch_feed = {self.input_state: states,
					  self.input_act: actions,
					  self.input_adv: advantages}
		loss, summary, _ = self.sess.run([self.loss, self.summary, self.train], feed_dict=batch_feed)
		self.writer.add_summary(summary, counter)

		return loss


	def get_action(self, state, x_mouse, y_mouse, mouse_pressed):
		def softmax(x):
			e_x = np.exp(x-np.max(x))
			return e_x / e_x.sum(axis=0)
		# possible action list:
		# 0 : mouseON and move + towards x axis
		# 1 : mouseON and move - towards x axis
		# 2 : mouseON and move + towards y axis
		# 3 : mouseON and move - towards y axis
		# 4 : mouse UP (resets xy position to the center of the slingshot)
		# 5 : do nothing
			    #Selecting actions without delay 
		#select random action
		xy_distance = 1

		#replicate state
		state = np.asarray([state], dtype=np.float32)/255.0
		# input_state = np.repeat(state, self.batch_size, axis=0)
		output = self.sess.run(self.actions, feed_dict={self.input_state: state})


		action = np.random.choice(self.action_num, p=output[0])

		# if epsilon > np.random.uniform(0,1):
		# 	action = np.argmax(output[0])
			
		# else:
		# 	action = np.random.randint(6, size=(1))[0]


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

		return action, x_mouse, y_mouse, mouse_pressed, softmax(output[0])


	def build_model(self):

		self.input_state = tf.placeholder(tf.float32, shape=self.input_shape)
		self.net, self.actions = self.Policy_Network(self.input_state)

		
		self.input_act = tf.placeholder(tf.float32, shape=[None, self.action_num])
		self.input_adv = tf.placeholder(tf.float32, shape=[None, 1])

		self.trainable_vars = tf.trainable_variables()

		# self.layers = []
		# for idx,net in enumerate(self.net):
		# 	self.layers.append(tf.summary.image("conv" + str(idx), net[:,:,:,:3]))

		print "Created slingshot model ..."

	
	def build_loss(self):
		#VERSION 1
		# self.log_prob = tf.log(self.actions)

		# # get log probs of actions from episode
		# indices = tf.range(0, tf.shape(self.log_prob)[0]) * tf.shape(self.log_prob)[1] + self.input_act
		# act_prob = tf.gather(tf.reshape(self.log_prob, [-1]), indices)
		# act_prob_f = tf.cast(act_prob, tf.float32)

		# # surrogate loss
		# self.loss = -tf.reduce_sum(tf.multiply(act_prob_f, self.input_adv))


		# self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='AdamOpt')
		# self.train = self.optimizer.minimize(self.loss, var_list=self.trainable_vars)



		#VERSION 2
		def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
		    discount_f = lambda a, v: a*self.gamma + v;
		    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
		    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
		    return tf_discounted_r

		tf_discounted_epr = tf_discount_rewards(self.input_adv)
		tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
		tf_discounted_epr -= tf_mean
		tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

		self.loss = tf.nn.l2_loss(self.input_act - self.actions)
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
		tf_grads = self.optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
		self.train = self.optimizer.apply_gradients(tf_grads)

		self.loss_graph = tf.summary.scalar("loss", self.loss)


		print "Created loss ..."


	def Policy_Network(self, input, name="policy_network"):

		with tf.variable_scope(name): 
			net = []

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
			fc1 = tf.contrib.layers.fully_connected(flattened, 512, 
													activation_fn=tf.nn.relu,
													scope='fc1')
			fc2 = tf.contrib.layers.fully_connected(fc1, 256, 
													activation_fn=tf.nn.relu,
													scope='fc2')
			fc3 = tf.contrib.layers.fully_connected(fc2, self.action_num, 
													activation_fn=None,
													scope='fc3')
			#implement fc layer
			output = tf.nn.softmax(fc3)

			return net, output
