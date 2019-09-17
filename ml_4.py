import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("C:/Users/User/Desktop/ml_course/datasets/howpop_train.csv")
test_df = pd.read_csv("C:/Users/User/Desktop/ml_course/datasets/howpop_test.csv")
print(train_df.T.head(), test_df.T.head(), train_df.columns, test_df.columns, sep='\n\n')
print(train_df.corr(), test_df.corr(), sep='\n\n')

'''for i, col1 in enumerate(train_df.corr().columns):
    for j in range(len(train_df.corr().columns)):
        if i != j:
            if train_df.corr()[col1].values[j] > .9:
                print('emelya')

for k, col2 in enumerate(test_df.corr().columns):
    for l in range(len(test_df.corr().columns)):
        if k != l:
            if test_df.corr()[col2].values[l] > .9:
                print('emelya')'''
train_df['published'] = pd.to_datetime(train_df['published'])
print(pd.DatetimeIndex(train_df['published']).year.value_counts())

features = ['author', 'flow', 'domain', 'title']
train_size = int(.7 * train_df.shape[0])
X, y = train_df.loc[:, features], train_df['favs_lognorm']
X_test = test_df.loc[:, features]
X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:, :]
y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]

tv = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))
X_train_title = tv.fit_transform(X_train['title'])
X_valid_title = tv.transform(X_valid['title'])
X_test_title = tv.transform(X_test['title'])
print(len(tv.vocabulary_))
print(tv.vocabulary_['python'])

tv_char_analyzer = TfidfVectorizer(analyzer='char')
X_train_title2 = tv_char_analyzer.fit_transform(X_train['title'])
X_valid_title2 = tv_char_analyzer.transform(X_valid['title'])
X_test_title2 = tv_char_analyzer.transform(X_test['title'])
print(tv_char_analyzer.vocabulary_, len(tv_char_analyzer.vocabulary_))

feats = ['author', 'flow', 'domain']
lal = X_train[feats][:5].fillna('-').T.to_dict().values()
dict_vect = DictVectorizer()
dict_vect_matrix = dict_vect.fit_transform(lal)
print(dict_vect_matrix, dict_vect_matrix.toarray(), sep='\n\n')
vectorizer_feats = DictVectorizer()
X_train_feats = vectorizer_feats.fit_transform(X_train[feats].fillna('-').T.to_dict().values())
X_valid_feats = vectorizer_feats.transform(X_valid[feats].fillna('-').T.to_dict().values())
X_test_feats = vectorizer_feats.transform(X_test[feats].fillna('-').T.to_dict().values())

X_train_new = scipy.sparse.hstack((X_train_title, X_train_title2, X_train_feats))
X_valid_new = scipy.sparse.hstack((X_valid_title, X_valid_title2, X_valid_feats))
X_test_new = scipy.sparse.hstack((X_test_title, X_test_title2, X_test_feats))

'''model1 = Ridge(alpha=0.1, random_state=1)
model1.fit(X_train_new, y_train)
train_pred1 = model1.predict(X_train_new)
valid_pred1 = model1.predict(X_valid_new)
print(mean_squared_error(y_train, train_pred1), mean_squared_error(y_valid, valid_pred1))

model2 = Ridge(alpha=1.0, random_state=1)
model2.fit(X_train_new, y_train)
train_pred2 = model2.predict(X_train_new)
valid_pred2 = model2.predict(X_valid_new)
print(mean_squared_error(y_train, train_pred2), mean_squared_error(y_valid, valid_pred2))
'''
model = Ridge(random_state=17)
train_data = scipy.sparse.vstack((X_train_new, X_valid_new))
model.fit(train_data, y)
print(mean_squared_error(y_valid, model.predict(X_valid_new)))