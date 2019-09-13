import pandas
from feature_analysis import get_feature_stats

from sklearn.datasets import load_boston
ds = load_boston()
df = pandas.DataFrame(ds['data'], columns=ds['feature_names'])
df = df.apply(lambda x: pandas.cut(x, bins=5, labels=['1','2','3','4','5']))

fs = get_feature_stats(df, ds['target'])