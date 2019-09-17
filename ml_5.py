import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

def delete_nun(data):
    for col in data.columns:
        data[col] = data[col].fillna(data[col].median())
    return data


def bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def interval(data, alpha):
    q = [100 * alpha / 2., 100 * (1 - alpha / 2.)]
    boundaries = np.percentile(data, q)
    return boundaries


data_frame = pd.read_csv("C:/Users/User/Desktop/ml_course/datasets/credit_scoring_sample.csv", sep=';')
print(data_frame.info(), data_frame.columns, data_frame.head(5), sep='\n\n')
df = delete_nun(data_frame)
print(df.head(5))
print(df['SeriousDlqin2yrs'].value_counts(normalize=True))
X, y = df.drop('SeriousDlqin2yrs', axis=1), df['SeriousDlqin2yrs']
foo = [df.columns[2], df.columns[4], df.columns[5]]
age_bootstrap = df[df['SeriousDlqin2yrs'] == 1]['age'].values
np.random.seed(0)
bts_mean = [np.mean(sample)
            for sample in bootstrap_samples(age_bootstrap, 1000)]
print(interval(bts_mean, 0.1))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

lr = LogisticRegression(class_weight='balanced', random_state=5)
parametrs = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
gs = GridSearchCV(lr, parametrs, scoring='roc_auc', cv=skf)
gs.fit(X, y)
print(gs.best_params_, gs.best_score_)


ss = StandardScaler()
model2 = LogisticRegression(C=0.001, class_weight='balanced', random_state=5)
model2.fit(ss.fit_transform(X), y)
coefs = model2.coef_.flatten().tolist()
tab = pd.DataFrame({'feat': X.columns, 'coef': coefs}).sort_values(by='coef', ascending=False)
print(tab)

'''rfc = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, class_weight='balanced')
params = {'max_depth': [5, 10, 15],
          'min_samples_leaf': [3, 5, 7, 9],
          'max_features': [1, 2, 4]}
forest_grid = GridSearchCV(rfc, params, cv=skf)
forest_grid.fit(X, y)
print(forest_grid.best_score_, forest_grid.best_params_)
print(forest_grid.best_score_ / gs.best_score_)
print(X.columns[np.argmin(forest_grid.best_estimator_.feature_importances_)])'''

params_v2 = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9],
             "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10, 100]}

bagg = BaggingClassifier(LogisticRegression(class_weight='balanced'), 100, random_state=42)
rs = RandomizedSearchCV(bagg, params_v2, n_iter=20, random_state=1, cv=skf, scoring='roc_auc    ')
rs.fit(X, y)
print(rs.best_score_, rs.best_params_)