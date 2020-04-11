# TODO import true model and datasets


def calc_acc_example():
    model = Model()
    accuracy = AccuracyScore()
    for X_batch, y_batch in dataset_gen:
        y_pred = model.predict(X_batch)
        accuracy(y_batch, y_pred)
    print(accuracy.score)


def measure_metric_example():
    model1 = Model()
    model2 = Model()
    dataset1 = []
    dataset2 = []

    argument_dict = [
        {
            'topic': "A-C",
            'load_model': lambda: model1,
            'load_dataset': lambda: dataset1
        },
        {
            'topic': "A-W",
            'load_model': lambda: model2,
            'load_dataset': lambda: dataset2
        }
    ]

    result = measure_metric(AccuracyScore, argument_dict)
