import pandas
from feature_analysis import get_feature_stats

from sklearn.datasets import load_boston
ds = load_boston()
df = pandas.DataFrame(ds['data'], columns=ds['feature_names'])
df = df.apply(lambda x: pandas.cut(x, bins=5, labels=['1','2','3','4','5']))

fs = get_feature_stats(df, ds['target'])



from sklearn.linear_model import LogisticRegression

models_to_try = {
    "sparse_pipeline1":Pipeline(steps=[('vect',vect), ('svd',TruncatedSVD()), ('lr',LogisticRegression())]),
    "sparse_pipeline3":Pipeline(steps=[('vect',vect), ('lr',LogisticRegression())])
}

params_to_try = {
    "sparse_pipeline1":{'svd__n_components':[10,100], 'lr__C':[0.1,1]},
    "sparse_pipeline3":{'lr__C':[0.1,1]}
}

model_features = {
    "sparse_pipeline1":(sparse_features, ova_train_df.target),
    "sparse_pipeline3":(sparse_features, ova_train_df.target)
}

model_parameters = (models_to_try, params_to_try, model_features)


best_model, metrics = get_optimal_model(model_parameters)
model_comparison = metrics.groupby("model").first()


