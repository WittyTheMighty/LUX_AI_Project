from chac import CompositeActorCritic

class SimpleCHAC(CompositeActorCritic):
    def __init__(self, env, config):
        super().__init__(SimpleCHAC, env, config)

    def predict(self, observation, unit_id):
        pass

    def subgoal_reached(self, sub_goal, observation):
        pass

    def observation_to_subgoal(self, observation):
        pass

    def compute_internal_reward(self, sub_goal, observation):
        pass

