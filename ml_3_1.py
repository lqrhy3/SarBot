import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def printf(*anything):
    print(anything, end='\n\n')


train_df = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/titanic_train.csv')
test_df = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/titanic_test.csv')
y = train_df['Survived']
printf(train_df.head(), test_df.head())
printf(train_df.info())
printf(train_df.describe(include='all'))
printf(test_df.describe(include='all'))

for col1 in train_df.columns:
    printf(train_df[col1].isnull().value_counts())
for col2 in test_df.columns:
    printf(test_df[col2].isnull().value_counts())

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Cabin'].fillna('G6', inplace=True)
train_df['Embarked'].fillna('S', inplace=True)
test_df['Cabin'].fillna('B57', inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
print(train_df.shape, train_df.dropna().shape, test_df.shape, test_df.dropna().shape)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Pclass'], prefix='PClass'),
                     pd.get_dummies(train_df['Sex'], prefix='Sex'),
                     pd.get_dummies(train_df['SibSp'], prefix='SibSp'),
                     pd.get_dummies(train_df['Parch'], prefix='Parch'),
                     pd.get_dummies(train_df['Embarked'], prefix='Embarked')], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['Pclass'], prefix='PClass'),
                     pd.get_dummies(test_df['Sex'], prefix='Sex'),
                     pd.get_dummies(test_df['SibSp'], prefix='SibSp'),
                     pd.get_dummies(test_df['Parch'], prefix='Parch'),
                     pd.get_dummies(test_df['Embarked'], prefix='Embarked')], axis=1)
train_df.drop(['PassengerId', 'Name', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Ticket', 'Cabin'],
              axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Ticket', 'Cabin'],
             axis=1, inplace=True)
print(set(test_df.columns) - set(train_df.columns))
test_df.drop(['Parch_9'], axis=1, inplace=True)

first_tree = DecisionTreeClassifier(max_depth=2, random_state=17)
model = first_tree.fit(train_df, y)
pred = first_tree.predict(test_df)
print(pred)

tree_params = {'max_depth': list(range(1, 5)),
               'min_samples_leaf': list(range(1, 5))}
grid = GridSearchCV(first_tree, tree_params, cv=5)
grid.fit(train_df, y)
print(grid.best_params_, grid.best_score_)