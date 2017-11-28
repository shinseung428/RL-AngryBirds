
import numpy as np
import tensorflow as tf
from architecture import *

class Agent():
	def __init__(self,config):

		self.gamma = .99               # discount factor for reward
		self.decay = 0.99    

		self.screen_w, self.screen_h, self.channel = (config.screen_w, config.screen_h, config.channel_dim)
		self.batch_size = config.batch_size

		self.input_shape = [None, self.screen_h, self.screen_w, self.channel]
		self.action_num = config.action_num
		
		self.learning_rate = config.learning_rate 
		self.momentum = config.momentum

		self.graphpath = config.graphpath
		self.modelpath = config.modelpath


		self.build_model()
		self.build_loss()

		self.summary = tf.summary.merge([self.loss_graph,self.layers])

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

		self.writer = tf.summary.FileWriter(self.graphpath, self.sess.graph)
		self.saver = tf.train.Saver()


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

		#replicate state
		state = np.asarray([state], dtype=np.float32)/255.0
		# input_state = np.repeat(state, self.batch_size, axis=0)
		output = self.sess.run(self.actions, feed_dict={self.input_state: state})

		action = np.random.choice(self.action_num, p=output[0])


		if action == 0:
			mouse_pressed = True
			x_mouse -= xy_distance
		elif action == 1:
			mouse_pressed = True
			y_mouse += xy_distance
		elif action == 2:
			mouse_pressed = False
			#reset the position of the mouse to the center point of the slingshot
			x_mouse, y_mouse = (135, 450)


		#bound the movement of the mouse
		if x_mouse < 60:
			x_mouse = 60
		if x_mouse > 250:
			x_mouse = 250
		if y_mouse < 370:
			y_mouse = 370
		if y_mouse > 520:
			y_mouse = 520

		return action, x_mouse, y_mouse, mouse_pressed, output[0]

	def update_model(self, states, actions, advantages, counter):
		#normalize input
		states = np.asarray(states, dtype=np.float32)/255.0

		batch_feed = {self.input_state: states,
					  self.input_act: actions,
					  self.input_adv: advantages}

		loss, summary, _ = self.sess.run([self.loss, self.summary, self.train], feed_dict=batch_feed)
		self.writer.add_summary(summary, counter)

		return loss

	def build_model(self):

		self.input_state = tf.placeholder(tf.float32, shape=self.input_shape)
		self.input_act = tf.placeholder(tf.int32, shape=[None, 1])
		self.input_adv = tf.placeholder(tf.float32, shape=[None, 1])

		self.net, self.actions = self.Policy_Network(self.input_state)

		self.trainable_vars = tf.trainable_variables()

		self.layers = []
		for idx,net in enumerate(self.net):
			self.layers.append(tf.summary.image("conv" + str(idx), net[:,:,:,:3]))

		print "Created slingshot model ..."

	
	def build_loss(self):
		def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
		    discount_f = lambda a, v: a*self.gamma + v;
		    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
		    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
		    return tf_discounted_r

		#discount rewards and normalize
		tf_discounted_epr = tf_discount_rewards(self.input_adv)
		tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
		tf_discounted_epr -= tf_mean
		tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

		# self.loss = tf.nn.l2_loss(self.input_act - self.actions)
		# self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
		# tf_grads = self.optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
		# self.train = self.optimizer.apply_gradients(tf_grads)


		#Another version 
		log_prob = tf.log(tf.clip_by_value(self.actions,1e-10,1.0))
		indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.input_act
		act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)


		self.loss = -tf.reduce_sum(tf.multiply(act_prob, tf_discounted_epr))

		self.loss = tf.Print(self.loss, [tf_discounted_epr], summarize = 400, message="discounted_epr: ")
		self.loss = tf.Print(self.loss, [self.input_act], summarize = 400, message="\input_act: ")
		self.loss = tf.Print(self.loss, [log_prob], summarize = 400, message="\nlog_prob: ")
		self.loss = tf.Print(self.loss, [act_prob], summarize = 400, message="\nact_prob: ")
		
		optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
		self.train = optimizer.minimize(self.loss)


		self.loss_graph = tf.summary.scalar("loss", self.loss)


		print "Created loss ..."


	def Policy_Network(self, input, name="policy_network"):

		with tf.variable_scope(name): 
			net = []

			conv1 = tf.contrib.layers.conv2d(input, 512, 5, stride=3, scope='conv1')
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

			flattened = tf.contrib.layers.flatten(conv4, scope='flattened')
			fc1 = tf.contrib.layers.fully_connected(flattened, 512, 
													activation_fn=tf.nn.relu,
													scope='fc1')
			#fc1 = tf.contrib.layers.batch_norm(fc1, scope='bn5')

			fc2 = tf.contrib.layers.fully_connected(fc1, 256, 
													activation_fn=tf.nn.relu,
													scope='fc2')
			#fc2 = tf.contrib.layers.batch_norm(fc2, scope='bn6')

			fc3 = tf.contrib.layers.fully_connected(fc2, self.action_num, 
													activation_fn=None,
													scope='fc3')
			#implement fc layer
			output = tf.nn.softmax(fc3)

			return net, output

	def save(self, num):
		save_path = self.saver.save(self.sess, self.modelpath + "slingshotmodel", global_step=num)

	def reload(self):
		latest_chkpt_path = tf.train.latest_checkpoint(self.modelpath)
		self.saver.restore(self.sess, latest_chkpt_path)
		print 'Reloaded model : ' + latest_chkpt_path

		game_steps = int(latest_chkpt_path.split('-')[1])
		return game_steps
