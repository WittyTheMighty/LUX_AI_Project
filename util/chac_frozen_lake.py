from chac import CompositeActorCritic


class SimpleCHAC(CompositeActorCritic):
    def __init__(self, env, config):
        super().__init__(SimpleCHAC, env, config)

    def subgoal_reached(self, sub_goal, observation):
        if sub_goal == observation:
            return True
        return False

    def observation_to_subgoal(self, observation):
        return observation

    def compute_internal_reward(self, sub_goal, observation):
        x1, y1 = self.env.to_coords(observation)
        x2, y2 = self.env.to_coords(sub_goal)
        dist = manhattan_dist(x1, y1, x2, y2)
        if dist < 0.5:
            return 1.0
        if dist < 2.0:
            return 0.5
        return 0.0


def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)
