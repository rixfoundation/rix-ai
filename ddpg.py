import numpy as np
from collections import deque
import random

from ActorCritic_network.actor_network import ActorNetwork
from ActorCritic_network.critic_network import CriticNetwork

REPLAY_MEMORY_SIZE = 300000  # play result = state of trade_board + selected action + reward + sign-off state
BATCH_SIZE = 256
GAMMA = 0.999
is_grad_inverter = False

class DDPG:
    # Deep Deterministic Policy Gradient Algorithm
    def __init__(self, session, currency, chart, timeline, length):
        self.currency = currency
        self.chart = chart
        self.timeline = timeline
        self.length = length
        self.num_action = currency

        self.critic_net = CriticNetwork(session, currency, chart, timeline, length)
        self.actor_net = ActorNetwork(session, currency, chart, timeline, length)

        # initialize buffer network:
        self.replay_memory = deque()

        # initialize time step:
        self.time_step = 0
        self.counter = 0

        action_boundary = [[-1, 1]] * currency

    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)

    def add_experience(self, state, next_state, action, reward, terminal):
        self.state = np.reshape(state, (self.currency * self.chart, self.timeline, self.length, 1))
        self.next_state = np.reshape(next_state, (self.currency * self.chart, self.timeline, self.length, 1))
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.replay_memory.append((self.state, self.next_state, self.action, self.reward, self.terminal))
        self.time_step = self.time_step + 1

        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

    def minibatches(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)

        # state t
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        #self.state_t_batch = np.reshape(self.state_t_batch, (self.currency * self.chart, self.timeline, self.length, 1))

        # state t+1
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array(self.state_t_1_batch)
        #self.state_t_1_batch = np.reshape(self.state_t_1_batch, (self.currency * self.chart, self.timeline, self.length, 1))

        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch, [len(self.action_batch), self.num_action])   # how define action_space?

        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)

        self.terminal_batch = [item[4] for item in batch]
        self.terminal_batch = np.array(self.terminal_batch)

    def train(self):
        # sample a random minibatch of N transitions from R
        self.minibatches()
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.state_t_1_batch)

        # Q'(s_(i+1), a_(i+1))
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_batch, self.action_t_1_batch)
        self.y_i_batch = []

        for idx in range(BATCH_SIZE):
            if self.terminal_batch[idx]:
                self.y_i_batch.append(self.reward_batch[idx])
            else:
                self.y_i_batch.append(self.reward_batch[idx] + GAMMA * q_t_1[idx][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])

        # update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_batch, self.action_batch, self.y_i_batch)

        # update actor proportional to the gradients
        action_for_deltaQ = self.evaluate_actor(self.state_t_batch)

        self.deltaQ_a = self.critic_net.compute_deltaQ_a(self.state_t_batch, action_for_deltaQ)[0]

        # train actor network proportional to deltaQ/delta_a and delta_actor_model/delta_actor_parameters:
        self.actor_net.train_actor(self.state_t_batch, self.deltaQ_a)

        # update target critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
