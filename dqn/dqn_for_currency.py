import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:

    REPLAY_MEMORY = 1000000   # play result = state of trade_board + selected action + reward + sign-off state
    BATCH_SIZE = 1024         # state value(=replay memory) for calculating on learning
    gamma = 0.9999            # reducing weight of past state
    STATE_LEN = 5           # number of frame reading at once (to consider past state)

    def __init__(self, session, currency, chart, timeline, length, n_action):
        self.session = session
        self.n_action = n_action
        self.currency = currency  # number of coin currency
        self.chart = chart      # number of chart
        self.length = length    # number of every contemporary info of coin currency
        self.timeline = timeline    # number of time series input in model

        self.memory = deque()   # memory the coin information from API saved
        self.state = None       # present state of coin info

        # variables input the state of game
        self.input_X = tf.placeholder(tf.float32, [None, currency * chart, timeline, length, self.STATE_LEN])    # data size, state frame
        self.input_A = tf.placeholder(tf.int64, [None])     # value of action which made each state
        self.input_Y = tf.placeholder(tf.float32, [None])   # value for loss function

        # network
        self.Q = self._build_network('main')
        self.target_Q = self._build_network('target')   # network calculating actually measured Q value
        self.cost, self.train_op = self._build_op()

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv3d(self.input_X, 32, [1, 1, 1], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv3d(model, 64, [1, 1, 1], padding='same', activation=tf.nn.relu)
            model = tf.layers.dense(tf.contrib.layers.flatten(model), 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    def _build_op(self):
        # gradient descent step on (y_j - Q(delta_j, a_j; theta))^2
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):    # main net -> target net
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])
        return action

    def init_state(self, state):    # initialize game trade_board state
        state = [state for _ in range(self.STATE_LEN)]
        #self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])
        self.state = np.stack(state, axis=3)

    def state_remember(self, state, action, reward, terminal):
        # state length = 5 (last one is the present)
        # when new state is input, the oldest one is eliminated so the length could be stayed 5
        next_state = np.reshape(state, (self.currency * self.chart, self.timeline, self.length, 1))
        next_state = np.append(self.state[:, :, :, 1:], next_state, axis=3)

        self.memory.append((self.state, next_state, action, reward, terminal))  # saving states in memory

        if len(self.memory) > self.REPLAY_MEMORY:   # restricted by 500000
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)     # sampling data about only batch size from memory

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        # Y value for loss function
        # Y = r_j if episode is terminated at j+1 step
        # Y = r_j + gamma*max(Q(delta_(j+1), a'; theta'))

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.gamma*np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X: state,
                                                   self.input_A: action,
                                                   self.input_Y: Y})
