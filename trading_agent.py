import numpy as np
import pandas as pd

from agent.Agent import Agent
from environment.EnvPlayer import PlayGround
from environment.Environment import TradeEnvironment

pair_name = "EURUSD"
interval = 1

dataframe = pd.read_csv("EURUSD1.mini.csv")
df = dataframe


class FxEnv(TradeEnvironment):

    @classmethod
    def __reward__(self, state, action):
        return -1 if action > 0 else 1


class FxTradeAgent(Agent):

    def act(self, state):
        return np.argmax(np.random.randint(1, 3, self.action_size))

    @classmethod
    def replay(self, memories):
        print(len(memories))


fx_env = FxEnv(df.values)

# print(state)

pl = PlayGround(env=fx_env, agent=FxTradeAgent(), time_frame=4)
pl.play()
