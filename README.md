# data_stream_gym

Use this simple lib for implement RL methods with few 

## Create environemt

```
class FxEnv(TradeEnvironment):

    @classmethod
    def __reward__(self, state, action):
        return -1 if action > 0 else 1
```
For futher development you need to implement your own Reward function

Overide 
def  __reward__(state,action)




## Create Agent

```

class FxTradeAgent(Agent):

    def act(self, state):
        return np.argmax(np.random.randint(1, 3, self.action_size))

    @classmethod
    def replay(self, memories):
        print(len(memories))

```

# Train algo

```
# Create Environment with data values
fx_env = FxEnv(df.values)
pl = PlayGround(env=fx_env, agent=FxTradeAgent(), time_frame=4)
pl.play()
```

## Licence
MIT

