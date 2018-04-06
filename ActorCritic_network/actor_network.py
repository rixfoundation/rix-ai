import tensorflow as tf

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001


class ActorNetwork:
    # Actor network model of DDPG(Deep Deterministic Policy Gradient) Algorithm
    #STATE_LEN = 5  # number of frame reading at once (to consider past state)

    def __init__(self, session, currency, chart, timeline, length):
        self.sess = session

        # variables input the state of trading, actor state does not be generated from actor_network.
        self.actor_state_in = tf.placeholder(tf.float32, [None,
                                                          currency * chart,
                                                          timeline,
                                                          length,#, self.STATE_LEN
                                                          1])  # data size, state frame

        # actor network model parameters
        self.W1_a, self.W2_a, self.W3_a, self.W4_a, self.W0_a, self.actor_model \
            = self.actor_network(currency, chart, length, timeline)

        # target network model parameters
        self.t_W1_a, self.t_W2_a, self.t_W3_a, self.t_W4_a, self.t_W0_a, self.t_actor_model \
            = self.actor_network(currency, chart, length, timeline)

        # cost of actor network:
        self.q_gradient_input = tf.placeholder(tf.float32, [None, currency]) # gets input from action_gradient computed in critic network file
        self.actor_parameters = [self.W1_a, self.W2_a, self.W3_a, self.W4_a, self.W0_a]
        self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.actor_parameters))

        self.sess.run(tf.global_variables_initializer())

        # To make sure actor & target have same initial parameters:
        self.sess.run([self.t_W1_a.assign(self.W1_a),
                       self.t_W2_a.assign(self.W2_a),
                       self.t_W3_a.assign(self.W3_a),
                       self.t_W4_a.assign(self.W4_a),
                       self.t_W0_a.assign(self.W0_a)])

        self.update_target_actor_op = [self.t_W1_a.assign(TAU * self.W1_a + (1 - TAU) * self.t_W1_a),
                                       self.t_W2_a.assign(TAU * self.W2_a + (1 - TAU) * self.t_W2_a),
                                       self.t_W3_a.assign(TAU * self.W3_a + (1 - TAU) * self.t_W3_a),
                                       self.t_W4_a.assign(TAU * self.W4_a + (1 - TAU) * self.t_W4_a),
                                       self.t_W0_a.assign(TAU * self.W0_a + (1 - TAU) * self.t_W0_a)]

    def actor_network(self, currency, chart, length, timeline):
        # Network that takes states and return action
        FILTER_1 = 16
        FILTER_2 = 32
        FILTER_3 = 64
        FILTER_4 = 128

        W1_a = tf.Variable(tf.random_normal([1, 1, 1, 1, FILTER_1], stddev=0.01))
        W2_a = tf.Variable(tf.random_normal([1, 1, 1, FILTER_1, FILTER_2], stddev=0.01))
        W3_a = tf.Variable(tf.random_normal([1, 1, 1, FILTER_2, FILTER_3], stddev=0.01))
        W4_a = tf.Variable(tf.random_normal([FILTER_3*currency*chart*length*timeline, FILTER_4]))
        W0_a = tf.Variable(tf.random_normal([FILTER_4, currency]))

        l1 = tf.nn.relu(tf.nn.conv3d(self.actor_state_in, W1_a, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.relu(tf.nn.conv3d(l1, W2_a, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.relu(tf.nn.conv3d(l2, W3_a, strides=[1, 1, 1, 1, 1], padding='SAME'))
        l3 = tf.reshape(l3, [-1, W4_a.get_shape().as_list()[0]])
        l4 = tf.nn.relu(tf.matmul(l3, W4_a))

        actor_model = tf.matmul(l4, W0_a)

        return W1_a, W2_a, W3_a, W4_a, W0_a, actor_model

    def evaluate_actor(self, state):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in: state})

    def evaluate_target_actor(self, t_state):
        return self.sess.run(self.t_actor_model, feed_dict={self.actor_state_in: t_state})

    def train_actor(self, actor_state_in, q_gradient_input):
        self.sess.run(self.optimizer, feed_dict={self.actor_state_in: actor_state_in,
                                                 self.q_gradient_input: q_gradient_input})

    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)
