class TradeEnvironment(object):

    def __init__(self, data):
        self.__data_feed = iter(data)
        self.pre_state = None

    @classmethod
    def __reward__(self, state, action):
        return 0

    def calculate_reward(self, state, action):
        reward = self.__reward__(state, action)
        return reward

    def get_next_state(self):
        '''
        :param last_action: action for the last state
        :return:
            reward - reward amount for your last action again last state
            state_next - next state
            done - True if the data stream end
        '''
        try:
            state_next = self.__data_feed.__next__()
        except:
            state_next = None
        done = True if state_next is None else False
        self.pre_state = state_next
        return state_next, done
