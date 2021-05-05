from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from functools import reduce

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

import matplotlib .pyplot as plt
import numpy as np
import pandas as pd


import warnings


class FeatureImportance:

        # Importance score = ROC AUC between each x from X and y
        def rocauc_importance(self,X,y):

            assert(len(set(y))==2), "ROC_AUC scoring works only with binary target variables"

            print("Calculating ROC AUC importance...")

            rocauc_results = []
            for feature in X.columns:
                rocauc_results.append((feature,roc_auc_score(y_true=y,y_score=X[feature])))

            result = pd.DataFrame(rocauc_results, columns=['feature','rocauc_imp'])
            result = result.set_index("feature")

            return result

        # Importance score = RandomForest embedded procedure
        def rf_importance(self,X,y):

            print("Calculating RandomForest importance...")

            rf = RandomForestClassifier(n_estimators=10, max_depth=8)
            rf.fit(X,y)
            result = pd.DataFrame(zip(X.columns, rf.feature_importances_), columns=['feature','rf_imp'])

            result['rf_imp_sortkey'] = result['rf_imp']
            result = result.sort_values(by=['rf_imp_sortkey'], ascending=False).drop(columns=['rf_imp_sortkey'])
            result = result.set_index("feature")

            return result

        # Importance score = GradientBoosting embedded procedure
        def gb_importance(self,X,y):

            print("Calculating Gradient Boosting importance...")

            gb = GradientBoostingClassifier(n_estimators=25)
            result = pd.DataFrame(zip(X.columns,gb.fit(X,y).feature_importances_),columns=['feature','gb_imp'])
            result['gb_imp_sortkey'] = result['gb_imp']
            result = result.sort_values(by=['gb_imp_sortkey'], ascending=False).drop(columns=['gb_imp_sortkey'])
            result = result.set_index("feature")

            return result

        # Importance score = Pearson correlation coefficient
        def corr_importance(self,X,y):

            warnings.warn("Using Pearson correlation with binary target", Warning, stacklevel=1)

            print("Calculating Pearson Correlation importance...")

            corr_result = []
            for feature in X.columns:
                corr_result.append(pearsonr(X[feature],y))
            result = pd.DataFrame(zip(X.columns, corr_result), columns=['feature','corr'])
            result['corr_imp'] = result['corr'].apply(lambda x: x[0])
            result['corr_pvalue'] = result['corr'].apply(lambda x: x[1])
            result = result.drop(columns=['corr'])

            result['corr_imp_sortkey'] = abs(result['corr_imp'])
            result = result.sort_values(by=['corr_imp_sortkey'], ascending=False).drop(columns=['corr_imp_sortkey'])
            result = result.set_index("feature")

            return result

        # Importance score = Mutual Information
        def mutual_info_importance(self,X,y):

            warnings.warn("Note that this method is resource greedy. It may take a lot of time to complete")

            print("Calculating Mutual Information importance...")       

            (X_sample,X_,y_sample,y_) = train_test_split(X, y, train_size=min(X.shape[0], 20000))

            result = pd.DataFrame({
                "feature":X.columns,
                "mi_imp":mutual_info_classif(X_sample,y_sample)
            })
            result['mi_imp_sortkey'] = abs(result['mi_imp'])
            result = result.sort_values(by=['mi_imp_sortkey'], ascending=False).drop(columns=['mi_imp_sortkey'])

            result = result.set_index("feature")

            return result

        # Compute all importances and get an aggregate rank
        def get_importances(self,X,y):

            rocauc_df = self.rocauc_importance(X,y)
            rf_df = self.rf_importance(X,y)
            gb_df = self.gb_importance(X,y)
            corr_df = self.corr_importance(X,y)[['corr_imp']]
            mi_df = self.mutual_info_importance(X,y)
            woe_df, iv_df = self.woe_importance(X,y)

            # Concatenate Horizontally
            result = reduce(lambda left,right: pd.merge(left,right,how='outer',on='feature'), [rocauc_df,rf_df,gb_df,corr_df,mi_df,iv_df])

            print(result)

            # Compute Ranks
            result['rocauc_rank'] = result['rocauc_imp'].rank(ascending=False)
            result['rf_rank'] = result['rf_imp'].rank(ascending=False)
            result['gb_rank'] = result['gb_imp'].rank(ascending=False)
            result['corr_rank'] = result['corr_imp'].rank(ascending=False)
            result['mi_rank'] = result['mi_imp'].rank(ascending=False)
            result['woe_importance'] = result['woe_imp'].rank(ascending=False)

            # Compute Aggregate Rank
            result['rank'] = 1.0*result.rocauc_rank + 1.0*result.rf_rank + 1.0*result.gb_rank + 0.5*result.corr_rank + 1.0*result.mi_rank

            return result

        def woe_importance(self,X,y):

            woe_table_list = []

            for feature in X.columns:

                base_table = pd.concat([X[[feature]],y], axis=1)
                base_table.columns = ['feature_category','target']
                base_table['n'] = 1

                stats_table = base_table.groupby(['feature_category','target'], as_index=False)['n'].sum()
                stats_table = stats_table.pivot(index='feature_category', columns='target',values='n')
                stats_table.columns=['neg','pos']
                stats_table['neg_recall'] = round(stats_table['neg'] / (stats_table['neg'].sum()+0.01),2)
                stats_table['pos_recall'] = round(stats_table['pos'] / (stats_table['pos'].sum()+0.01),2)
                stats_table['neg_prec'] = round(stats_table['neg'] / (stats_table['pos'] + stats_table['neg']),2)
                stats_table['pos_prec'] = round(stats_table['pos'] / (stats_table['pos'] + stats_table['neg']),2)
                stats_table['woe'] = np.log((stats_table['pos_recall']+0.01)/(stats_table['neg_recall'] + 0.01))
                stats_table['woe_imp'] = (stats_table['pos_recall'] - stats_table['neg_recall']) * stats_table['woe']

                stats_table['feature'] = feature
                stats_table = stats_table.reset_index()
                stats_table = stats_table[['feature'] + list(stats_table.columns[:-1])]

                woe_table_list.append(stats_table)

            woe_table = pd.concat(woe_table_list)

            return (woe_table, woe_table.groupby('feature')['woe_imp'].sum())


        def save_png(self, df_imp, filename):

            print("Saving PNG to disk...")

            plt.figure(figsize=(10,10))

            df_to_draw = df_imp[['rank']].reset_index().sort_values("rank")[0:25]
            df_to_draw = df_to_draw.sort_values("rank", ascending=False)

            plt.barh(
                df_to_draw.feature, 
                width = df_to_draw['rank'].max() - df_to_draw['rank'],
                height = 0.75,
                alpha=0.5, 
                edgecolor='black')

            plt.grid(True, axis='y')

            plt.savefig("{}.png".format(filename), dpi=300)

            return -1
