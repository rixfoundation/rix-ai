import tensorflow as tf

TAU = 0.001
LEARNING_RATE = 0.00001
BATCH_SIZE = 256


class CriticNetwork:
    # Criic Q value model of the DDPG algorithm
    #STATE_LEN = 5  # number of frame reading at once (to consider past state)

    def __init__(self, session, currency, chart, timeline, length):
        self.sess = session

        # variables input the state of trading, critic state and action do not be generated from critic network.
        self.critic_state_in = tf.placeholder(tf.float32, [None,
                                                           currency * chart,
                                                           timeline,
                                                           length,#, self.STATE_LEN
                                                           1])  # data size, state frame
        self.critic_action_in = tf.placeholder(tf.float32, [None, currency])  # value of action which made each state
        self.q_value_in = tf.placeholder(tf.float32, [None, 1])  # supervisor: value for loss function

        # critic_q_model parameters:
        self.W1_c, self.W2_c, self.W3_c, self.W4_c, self.W4_action_c, self.W0_c, self.critic_q_model \
            = self.critic_network(currency, chart, length, timeline)

        # create_target_q_model:
        self.t_W1_c, self.t_W2_c, self.t_W3_c, self.t_W4_c, self.t_W4_action_c, self.t_W0_c, self.t_critic_q_model \
            = self.critic_network(currency, chart, length, timeline)

        self.l2_regularizer_loss = 0.0001 * tf.reduce_sum(tf.square(self.W4_c))
        self.cost = tf.square(self.critic_q_model - self.q_value_in) / BATCH_SIZE + self.l2_regularizer_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

        # from simple actor network:
        self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
        self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]    #this is just divided by batch size

        self.sess.run(tf.global_variables_initializer())

        # to make sure critic and target have same parameters copy the parameters
        self.sess.run([self.t_W1_c.assign(self.W1_c),
                       self.t_W2_c.assign(self.W2_c),
                       self.t_W3_c.assign(self.W3_c),
                       self.t_W4_c.assign(self.W4_c),
                       self.t_W4_action_c.assign(self.W4_action_c),
                       self.t_W0_c.assign(self.W0_c),])

        self.update_target_critic_op = [self.t_W1_c.assign(TAU * self.W1_c + (1 - TAU) * self.t_W1_c),
                                        self.t_W2_c.assign(TAU * self.W2_c + (1 - TAU) * self.t_W2_c),
                                        self.t_W3_c.assign(TAU * self.W3_c + (1 - TAU) * self.t_W3_c),
                                        self.t_W4_c.assign(TAU * self.W4_c + (1 - TAU) * self.t_W4_c),
                                        self.t_W4_action_c.assign(TAU * self.W4_action_c + (1 - TAU) * self.t_W4_action_c),
                                        self.t_W0_c.assign(TAU * self.W0_c + (1 - TAU) * self.t_W0_c),]

    def critic_network(self, currency, chart, length, timeline):
        FILTER_1 = 16
        FILTER_2 = 32
        FILTER_3 = 64
        FILTER_4 = 128

        W1_c = tf.Variable(tf.random_normal([1, 1, 1, 1, FILTER_1], stddev=0.01))
        W2_c = tf.Variable(tf.random_normal([1, 1, 1, FILTER_1, FILTER_2], stddev=0.01))
        W3_c = tf.Variable(tf.random_normal([1, 1, 1, FILTER_2, FILTER_3], stddev=0.01))
        W4_c = tf.Variable(tf.random_normal([FILTER_3*currency*chart*length*timeline, FILTER_4]))
        W4_action_c = tf.Variable(tf.random_normal([currency, FILTER_4]))
        W0_c = tf.Variable(tf.random_normal([FILTER_4, currency]))

        l1 = tf.nn.relu(tf.nn.conv3d(self.critic_state_in, W1_c, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.relu(tf.nn.conv3d(l1, W2_c, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.relu(tf.nn.conv3d(l2, W3_c, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l3 = tf.reshape(l3, [-1, W4_c.get_shape().as_list()[0]])
        action_in = tf.reshape(self.critic_action_in, [-1, W4_action_c.get_shape().as_list()[0]])
        l4 = tf.nn.relu(tf.matmul(l3, W4_c) + tf.matmul(action_in, W4_action_c))

        critic_q_model = tf.matmul(l4, W0_c)

        return W1_c, W2_c, W3_c, W4_c, W4_action_c, W0_c, critic_q_model

    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch,
                                                 self.critic_action_in: action_batch,
                                                 self.q_value_in: y_i_batch})

    def evaluate_target_critic(self, state_t_1, action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.critic_state_in: state_t_1,
                                                               self.critic_action_in: action_t_1})

    def compute_deltaQ_a(self, state_t, action_t):
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t,
                                                               self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
