from collections import deque

import random


class Agent(object):

    def __init__(self,
                 action_size=3,
                 max_length=1000, replay_prob=0.995, forget_rate=0.25):
        self.action_size = action_size
        self.forget_rate = forget_rate
        self.replay_prob = replay_prob
        self.max_length = max_length
        self.__memory__ = deque(maxlen=self.max_length)
        self.after_init()

    @classmethod
    def after_init(self):
        pass

    @classmethod
    def act(self, state):
        return -1

    @classmethod
    def replay(self, memories):
        pass

    @classmethod
    def after_memories(self, train_status):
        pass

    def memorize(self, state_t, action_t, reward_t, state_t_next, done):
        self.__memory__.append((state_t, action_t, reward_t, state_t_next, done))
        evaluate_request = False
        if len(self.__memory__) == self.__memory__.maxlen:
            if random.uniform(0, 1) <= self.replay_prob:
                self.replay(self.__memory__)

                # Randomly forgot memories
                for f in range(int(len(self.__memory__) * (random.uniform(0, self.forget_rate)))):
                    self.__memory__.pop()
                print("Left Memeory Length {}".format(len(self.__memory__)))

            self.__memory__.pop()  # forgot first
            evaluate_request = True
            self.after_memories(evaluate_request)

        return evaluate_request
