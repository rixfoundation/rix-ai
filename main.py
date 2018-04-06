import tensorflow as tf
import numpy as np
import datetime

from ddpg import DDPG
from trade_board import setting

import train_test_data

#is_batch_norm = False   # batch normalization switch... but do i use?

tf.app.flags.DEFINE_boolean('train', True, 'on learning mode.')
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 10000
TRAIN_INTERVAL = 5
OBSERVE = 1000      # must larger than batch_size (every batch should be chosen in observed state)

#act_species = setting.act_species
#NUM_ACTION = act_species * len(setting.currency)
CURRENCY = len(setting.currency)
CHART = len(setting.candle_state)
LENGTH = len(setting.parameters)
TIMELINE = setting.timeline       # same as in setting.py


def main():
    sess = tf.Session()

    setting.load_data(setting.currency, train_test_data.file_list, train_test_data.test_file)
    agent = DDPG(sess, CURRENCY, CHART, TIMELINE, LENGTH)
    counter = 0
    reward_for_episode = 0
    total_reward = 0

    epsilon = 1.0       # parameter defining ratio between random action and DQN decision
    time_step = 0       # frame number

    # saving reward
    reward_st = np.array([0])

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./trade_model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('model has been loaded successfully!')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('start new progress.')
        sess.run(tf.global_variables_initializer())

    for idx in range(MAX_EPISODE):
        terminal = False
        print('Starting episode no: %d' % idx)
        state = setting.reset()
        reward_for_episode = 0
        step_on_episode = 0

        while not terminal:
            present_state = state
            if np.random.rand() < epsilon:
                selected_currency = np.random.choice(CURRENCY)
                ratio = 2 * (np.random.rand() - 0.5)
                action = setting.action_value(CURRENCY, selected_currency, ratio)

            else:
                action = agent.evaluate_actor(present_state)

            if idx > OBSERVE:
                epsilon -= 1 / 50000

            state, reward, terminal, _ = setting.step(action)

            # add s_t, s_(t+1), action, reward to experience memory
            agent.add_experience(present_state, state, action, reward, terminal)

            # train critic and actor network
            if time_step > 2000 and time_step % TRAIN_INTERVAL == 0:
                agent.train()

            reward_for_episode += reward
            time_step += 1
            step_on_episode += 1

        # check if episode ends
        print('at %s, EPISODE: %d, Steps: %d, Reward: %d' %
              (str(datetime.datetime.now()), idx, step_on_episode, reward_for_episode))
        reward_st = np.append(reward_st, reward_for_episode)

        if idx % 500 == 0 and idx != 0:
            saver.save(sess, 'trade_model/actor_critic_network.ckpt', global_step=time_step)

    total_reward += reward_for_episode
    print('Average reward per episode: {}'.format(total_reward / MAX_EPISODE))


if __name__ == '__main__':
    main()

