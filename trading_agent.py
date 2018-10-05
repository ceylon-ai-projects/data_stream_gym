import numpy as np

from agent.Agent import Agent
from data.data_manager import get_data_chunk
from environment.EnvPlayer import PlayGround
from environment.Environment import TradeEnvironment
from preprocess.data_pre_process import create_data_frame

import tensorflow as tf

obs_length = 21
action_size = 3

# Observations Count
observations = tf.placeholder(shape=[1, obs_length], dtype=tf.float32)
# 0,1,2 BUY, STAY, SELL
actions = tf.placeholder(shape=[None], dtype=tf.int32)
# +1, -1 with discount
rewards = tf.placeholder(shape=[None], dtype=tf.float32)

# model
Y = tf.layers.dense(observations, 200, activation=tf.nn.relu)
Ylogits = tf.layers.dense(Y, action_size)

# Sample an action from predicted probabilities
sample_op = tf.multinomial(logits=tf.reshape(Ylogits, shape=(1, 3)), num_samples=1)

# loss
cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions, action_size),
                                                  logits=Ylogits)
loss = tf.reduce_sum(rewards * cross_entropies)

# Training Operation
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99)
train_op = optimizer.minimize(loss)

init_op = tf.initialize_all_variables()


def dif_to_action(diff):
    if diff < 0:
        return 0  # Sell
    elif diff == 0:
        return 1  # Stay
    else:
        return 2  # Buy


class FxEnv(TradeEnvironment):

    @classmethod
    def __reward__(self, state, action, state_t):
        diff = state_t[:1] - state[:1]
        actual_action = dif_to_action(diff)
        if actual_action - action == 0:
            return 1
        else:
            return -1


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()


class FxTradeAgent(Agent):

    @classmethod
    def after_init(self):
        self.sess = tf.Session()
        self.sess.run(init_op)

    def act(self, state):
        if state is not None:
            state = np.reshape(state, (1, obs_length))
            return self.sess.run(sample_op, feed_dict={observations: state})
        return np.argmax(np.random.randint(1, 3, self.action_size))

    @classmethod
    def replay(self, memories):
        for state_t_pre, action_t_pre, reward_t_pre, state_t, done in memories:
            action_t_pre = np.array(action_t_pre)
            reward_t_pre = np.array(reward_t_pre)
            state_t_pre = np.array(state_t_pre)

            # Reshape Inputs
            action_t_pre = np.reshape(action_t_pre, (action_t_pre.shape[0],))
            reward_t_pre = np.reshape(reward_t_pre, (1,))
            state_t_pre = np.reshape(state_t_pre, (1, state_t_pre.shape[0]))

            # print(action_t_pre.shape, action_t_pre.shape, action_t_pre.shape)

            feed_dict = {
                rewards: reward_t_pre,
                observations: state_t_pre,
                actions: action_t_pre
            }
            self.sess.run(train_op, feed_dict=feed_dict)


pair_name = "EURUSD"
interval = 1

future_state = 4
state_size = 47
action_size = 3
considering_steps = 15

rsi_range = [14]
tsi_range = [14, 29, 58, 100]
emi_range = [3, 89]
aroon_range = [3, 21, 89]
dpo_range = [3, 21, 89]

chunk_size = 2e5

fx_agent = FxTradeAgent(max_length=100)

data_frames = get_data_chunk(pair_name, interval, chunk_size=10000)

for data_frame in data_frames:
    print("\n----Start Processing Another Chunk of Data ----")
    print(data_frame.head(1))
    print(data_frame.tail(1))
    print("----")
    df = create_data_frame(data_frame,
                           considering_steps=considering_steps,
                           rsi_range=rsi_range,
                           tsi_range=tsi_range,
                           emi_range=emi_range,
                           aroon_range=aroon_range,
                           dpo_range=dpo_range)
    print("---Data Summary---")
    print(df.head())
    print(df.tail())
    print(f"Before Process {len(data_frame)}")
    print(f"After Process {len(df)}")
    print("\n")
    fx_env = FxEnv(df.values)

    # print(state)
    pl = PlayGround(env=fx_env, agent=fx_agent, time_frame=1)
    pl.play()
