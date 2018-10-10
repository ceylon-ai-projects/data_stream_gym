import numpy as np

from agent.Agent import Agent
from data.data_manager import get_data_chunk
from environment.EnvPlayer import PlayGround
from environment.Environment import TradeEnvironment
from preprocess.data_pre_process import create_data_frame

import tensorflow as tf
import os

dirname = os.path.dirname(__file__)
base_path = dirname + "/model_saved"
model_path = "{}/model".format(base_path)

base_path = dirname + "/logs"
log_path = "{}".format(base_path)

print("Model path => {}".format(model_path))
print("Log path => {}".format(log_path))

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
Y = tf.layers.dense(Y, 100, activation=tf.nn.relu)
Y = tf.layers.dense(Y, 50, activation=tf.nn.relu)
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

saver = tf.train.Saver()


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
        if state_t is not None and state is not None:
            diff = state_t[:1] - state[:1]
            actual_action = dif_to_action(diff)
            if actual_action - action == 0:
                return 1
            else:
                return -1
        return 0


class FxTradeAgent(Agent):
    epsilon = 0.5
    epsilon_decay = 0.995
    train_agent = True

    @classmethod
    def after_init(self):
        self.sess = tf.Session()

        saver.restore(self.sess, model_path)
        # self.sess.run(init_op)
        file_writer = tf.summary.FileWriter(log_path, self.sess.graph)

    def get_policy_decision(self, state):
        if state is not None:
            state = np.reshape(state, (1, obs_length))
            return self.sess.run(sample_op, feed_dict={observations: state})
        return np.argmax(np.random.randint(1, 3, self.action_size))

    def act(self, state):
        # Act with epslion on traning process
        if self.train_agent is False:
            return self.get_policy_decision(state)
        else:
            if np.random.rand() >= self.epsilon:
                return self.get_policy_decision(state)
            else:
                return np.argmax(np.random.randint(1, 3, self.action_size))

    def after_memories(self, train_status):
        if train_status:
            self.epsilon = self.epsilon * self.epsilon_decay

    def replay(self, memories):

        for state_t_pre, action_t_pre, reward_t_pre, state_t, done in memories:
            # if action_t_pre !=0:
            #     print(action_t_pre)

            action_t_pre = np.array(action_t_pre)
            reward_t_pre = np.array(reward_t_pre)
            state_t_pre = np.array(state_t_pre)
            # Reshape Inputs
            action_t_pre = np.reshape(action_t_pre, (1,))
            reward_t_pre = np.reshape(reward_t_pre, (1,))
            state_t_pre = np.reshape(state_t_pre, (1, state_t_pre.shape[0]))

            # print(state_t_pre, action_t_pre, reward_t_pre, state_t)

            feed_dict = {
                rewards: reward_t_pre,
                observations: state_t_pre,
                actions: action_t_pre
            }
            self.sess.run(train_op, feed_dict=feed_dict)

        saver.save(self.sess, save_path=model_path)

        with tf.name_scope('cross_entropy'):
            tf.summary.scalar('cross_entropy', cross_entropies)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Ylogits, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        # file_writer = tf.summary.FileWriter(log_path, self.sess.graph)
        file_writer = tf.summary.FileWriter(log_path)


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

chunk_size = 2e4

fx_agent = FxTradeAgent(max_length=20000)

data_frames = get_data_chunk(pair_name, interval,
                             chunk_size=chunk_size)

playground_step = 0

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
    pl = PlayGround(env=fx_env,
                    agent=fx_agent,
                    time_frame=1,
                    playground_step=playground_step)
    pl.play()
    playground_step += 1
