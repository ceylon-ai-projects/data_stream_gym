class PlayGround(object):
    reward_history = []

    def __init__(self, env, agent, time_frame=4):
        self.time_frame = time_frame
        self.env = env
        self.agent = agent

    def play(self):
        done = False
        state_t_pre = None
        action_t_pre = None
        steps = 0
        agent_act = False
        while done is False:
            state_t, done = self.env.get_next_state()

            if steps % self.time_frame == 0:
                if agent_act:
                    reward_t_pre = self.env.calculate_reward(state_t_pre, action_t_pre, state_t)
                    # Recode history

                    # print(state_t_pre[:1], action_t_pre, state_t[:1], reward_t_pre)

                    self.reward_history.append((steps, reward_t_pre))
                    self.agent.memorize(state_t_pre, action_t_pre, reward_t_pre, state_t, done)
                    agent_act = False

                if agent_act is False:
                    action_t = self.agent.act(state_t)
                    agent_act = True
                    action_t_pre = action_t
                    state_t_pre = state_t

            steps += 1
