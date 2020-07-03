params = {
    'env': {
        'env': 'microgrid',
        'tr_st_date': '2016-01-01',  # train starting date
        'tr_en_date': '2016-01-03',  # train ending date
        'case': 'elespino_continuous',
        'te_st_date': '2016-01-01',  # test starting date
        'te_en_date': '2017-07-31'   # test ending date
    },
    'problem': {
        'T': 100,
        'nb_train_steps': 48000,
        'nb_test_steps': 10000
    }
}
