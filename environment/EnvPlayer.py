from app_worker import write_data_hist


class PlayGround(object):
    reward_history = []

    def __init__(self, env, agent, time_frame=4, playground_step=0):
        self.playground_step = playground_step
        self.time_frame = time_frame
        self.env = env
        self.agent = agent

    def play(self):
        done = False
        state_t_pre = None
        action_t_pre = None
        steps = 0
        agent_act = False
        reward_recodes = 0
        while done is False:
            state_t, done = self.env.get_next_state()

            step_ = self.playground_step + steps

            if steps % self.time_frame == 0:
                if agent_act:
                    reward_t_pre = self.env.calculate_reward(state_t_pre, action_t_pre, state_t)

                    # Recode history

                    # print(state_t_pre[:1], action_t_pre, state_t[:1], reward_t_pre)
                    # print(step_)

                    if step_ % 10 == 0:
                        write_data_hist.delay(step_, reward_recodes / steps)
                    reward_recodes += reward_t_pre
                    self.agent.memorize(state_t_pre, action_t_pre, reward_t_pre, state_t, done)
                    agent_act = False

                if agent_act is False:
                    action_t = self.agent.act(state_t)
                    # print(action_t)
                    agent_act = True
                    action_t_pre = action_t
                    state_t_pre = state_t

            steps += 1
