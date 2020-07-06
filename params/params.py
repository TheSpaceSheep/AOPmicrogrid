params = {
    'env': {
        'env': 'microgrid',
        'tr_st_date': '2016-01-01',  # train starting date
        'tr_en_date': '2017-01-03',  # train ending date
        'te_st_date': '2016-01-01',  # test starting date
        'te_en_date': '2017-07-31',   # test ending date
        'case': 'elespino_continuous'
    },
    'problem': {
        'T': 100,
        'nb_train_steps': 1000,
        'nb_test_steps': 10000
    },
    'agent': {
        'ppo': {
            'policy': 'MlpPolicy',
            'gamma': 0.99,
            'n_steps': 64,
            'ent_coef': 0.01,
            'learning_rate': 0.00025,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'lam': 0.95,
            'nminibatches': 4,
            'noptepochs': 4,
            'cliprange': 0.2,
            'tensorboard': 'tensorboard/'
        },
        'dqn': {
            'gamma': 0.99,
            'learning_rate': 0.0005,
            'buffer_size': 50000,
            'exploration_fraction': 0.1,
            'train_freq': 1,
            'batch_size': 32,
            'double_q': True,
            'learning_starts': 1000,
            'target_network_update_freq': 500,
            'prioritized_replay': True,
            'tensorboard': 'tensorboard/'
        }
        'mpc': {
            'H': 10
        }
    }
}
