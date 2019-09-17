import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/adult_train.csv', sep=';')
df_test = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/adult_test.csv', sep=';')
print(df_train.head(5), df_test.head(5))
df_test = df_test[(df_test['Target'] == ' >50K.') | (df_test['Target'] == ' <=50K.')]
df_train.at[df_train['Target'] == ' <=50K', 'Target'] = 0
df_train.at[df_train['Target'] == ' >50K', 'Target'] = 1
df_test.at[df_test['Target'] == ' <=50K.', 'Target'] = 0
df_test.at[df_test['Target'] == ' >50K.', 'Target'] = 1
print(df_train.head(5), df_test.head(5), sep='\n\n')

print(df_train.dtypes)
print(df_test.dtypes)
df_test['Age'] = df_test['Age'].astype('int64')
df_test['fnlwgt'] = df_test['fnlwgt'].astype('int64')
df_test['Education_Num'] = df_test['Education_Num'].astype('int64')
df_test['Capital_Gain'] = df_test['Capital_Gain'].astype('int64')
df_test['Capital_Loss'] = df_test['Capital_Loss'].astype('int64')
df_test['Hours_per_week'] = df_test['Hours_per_week'].astype('int64')
print(df_test.dtypes)
categ_feat_train = [c for c in df_train.columns if df_train[c].dtype == 'object']
num_feat_train = [c for c in df_train.columns if df_train[c].dtype == 'int64']
categ_feat_test = [c for c in df_test.columns if df_test[c].dtype == 'object']
num_feat_test = [c for c in df_test.columns if df_test[c].dtype == 'int64']
print(categ_feat_train, num_feat_train, categ_feat_test, num_feat_test, sep='\n\n')
for feat in categ_feat_train:
    df_train[feat] = df_train[feat].fillna(df_train[feat].mode())

for feat in categ_feat_test:
    df_test[feat] = df_test[feat].fillna(df_test[feat].mode())

for feat in num_feat_train:
    df_train[feat] = df_train[feat].fillna(df_train[feat].median())

for feat in num_feat_test:
    df_test[feat] = df_test[feat].fillna(df_test[feat].median())
print(df_train.shape, df_train.dropna().shape, df_test.shape, df_test.dropna().shape)
df_train = pd.concat([df_train, pd.get_dummies(df_train['Workclass'], prefix="Workclass"),
                        pd.get_dummies(df_train['Education'], prefix="Education"),
                        pd.get_dummies(df_train['Martial_Status'], prefix="Martial_Status"),
                        pd.get_dummies(df_train['Occupation'], prefix="Occupation"),
                        pd.get_dummies(df_train['Relationship'], prefix="Relationship"),
                        pd.get_dummies(df_train['Race'], prefix="Race"),
                        pd.get_dummies(df_train['Sex'], prefix="Sex"),
                        pd.get_dummies(df_train['Country'], prefix="Country")], axis=1)

df_test = pd.concat([df_test, pd.get_dummies(df_test['Workclass'], prefix="Workclass"),
                      pd.get_dummies(df_test['Education'], prefix="Education"),
                      pd.get_dummies(df_test['Martial_Status'], prefix="Martial_Status"),
                      pd.get_dummies(df_test['Occupation'], prefix="Occupation"),
                      pd.get_dummies(df_test['Relationship'], prefix="Relationship"),
                      pd.get_dummies(df_test['Race'], prefix="Race"),
                      pd.get_dummies(df_test['Sex'], prefix="Sex"),
                      pd.get_dummies(df_test['Country'], prefix="Country")],
                      axis=1)

df_train.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'],
              axis=1, inplace=True)
df_test.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'],
             axis=1, inplace=True)
print(df_train.shape, df_test.shape)
print(set(df_train.columns) - set(df_test.columns))
df_train.drop(['Country_ Holand-Netherlands'], axis=1, inplace=True)
print(df_train.shape, df_test.shape)

X_train = df_train.drop(['Target'], axis=1)
y_train = df_train['Target']
X_test = df_test.drop(['Target'], axis=1)
y_test = df_test['Target']

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
print(accuracy_score(y_test, tree_predictions))

tree_params = {'max_depth': range(1, 11)}
locally_best_tree = GridSearchCV(tree, tree_params, cv=5)
locally_best_tree.fit(X_train, y_train)
print(locally_best_tree.best_params_, locally_best_tree.best_score_)
tree_best = DecisionTreeClassifier(max_depth=9, random_state=17)
tree_best.fit(X_train, y_train)
tree_best_pred = tree_best.predict(X_test)
print(accuracy_score(y_test, tree_best_pred))
