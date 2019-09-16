from sklearn.model_selection import GridSearchCV
import pandas

def get_optimal_model(model_parameters):

    models_to_try, params_to_try, model_features = model_parameters[0], model_parameters[1], model_parameters[2]
    model_metrics = pandas.DataFrame(columns=['params','quality','std'])
    model_metrics = []

    for i, current_model_name in enumerate(models_to_try.keys()):

        current_model = models_to_try[current_model_name]
        print("Processing model = {}".format(current_model))
        param_map = params_to_try[current_model_name]
        gridsearch = GridSearchCV(estimator = current_model, param_grid = param_map, scoring='roc_auc', cv = 5, n_jobs=-1, verbose=0)
        X = model_features[current_model_name][0]
        y = model_features[current_model_name][1]
        gridsearch.fit(X, y)

        model = gridsearch.best_estimator_
        cv_results = gridsearch.cv_results_

        current_model_metrics = pandas.DataFrame.from_records(list(zip(cv_results['params'], cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['mean_fit_time'])), columns=['params','quality','std','fit_time'])
        current_model_metrics['model'] = current_model_name
        model_metrics.append(current_model_metrics)

    metrics = pandas.concat(model_metrics)

    metrics['lb'] = metrics['quality'] - 2 * metrics['std']
    metrics['ub'] = metrics['quality'] + 2 * metrics['std']

    return (model, metrics.sort_values("quality", ascending=False))