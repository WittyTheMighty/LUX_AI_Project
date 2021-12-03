from Lux_Project_Env import frozen_lake

if __name__ == '__main__':
    env = frozen_lake.FrozenLakeEnv(is_slippery=False)

    config_PPO = {
        'actor_lr': 0.0003,
        'critic_lr': 0.0005,
        'action_dim': 4,
        'state_dim': 1,
        'subgoal_dim': 1,
        "gamma": 0.99,
        "eps_clip": 0.2,
        'K_epochs': 10,
        'has_continuous_action_space': False,
        'action_std_init':0.6,
    }

    config = {
        'state_dim':1,
        'state_n' : 1,
        'subgoal_dim': 1,
        'subgoal_n' : 1,
        'std_dev':0.2,
        'critic_lr': 0.002,
        'actor_lr': 0.001,
        'gamma' : 0.99,
        'tau': 0.005,
    }