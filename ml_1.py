import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/adult.data.csv')

print(df.head(), df.columns, end='\n\n')

print(df['sex'].value_counts(), end='\n\n')

print(df[df['sex'] == 'Female']['age'].mean(), end='\n\n')

print(df['native-country'].value_counts(normalize=True), end='\n\n')

print(df[df['salary'] == '<=50K']['age'].describe(), end='\n\n')

print(df[df['salary'] == '>50K']['age'].describe(), end='\n\n')

print(df[(df['education'] > 'Assoc-voc') & (df['salary'] == '>50K')].shape[0] == df[(df['education'] > 'Assoc-voc')].shape[0], end='\n\n')

print(df.groupby(['race'])['age', 'sex'].describe(), end='\n\n')

print(df.groupby(['sex', 'race'])['age'].describe(), end='\n\n')

print(pd.crosstab(df['marital-status'], df['salary']), end='\n\n')


def married_status(marital_status):
    if marital_status == 'Married-AF-spouse' or\
       marital_status == 'Married-civ-spouse' or\
       marital_status == 'Married-spouse-absent':
        return True
    else:
        return False


df['married'] = df['marital-status'].map(married_status)
print(df[df['salary'] == '>50K']['married'].value_counts(normalize=True), end='\n\n')

print(df['hours-per-week'].max(), end='\n\n')

print(df[df['hours-per-week'] == 99].shape[0], end='\n\n')

print(df[df['hours-per-week'] == 99]['salary'].value_counts(normalize=True), end='\n\n')

print(df.groupby(['native-country', 'salary'])['hours-per-week'].describe())

df2 = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/titanic_train.csv')
print(df2.head(), df2.info(), sep='\n\n', end='\n\n')
print(df2['Sex'].value_counts())
print(df2['Fare'].describe())
print(df2[df2['Age'] < 30.0]['Survived'].value_counts(normalize=True))
print(df2[df2['Age'] > 60.0]['Survived'].value_counts(normalize=True))
print(df2['Name'].value_counts())


def first_name(name):
    name = name.split()
    for i in range(len(name)):
        if name[i] == 'Mr.':
            return name[i+1]


df2['freq'] = df2['Name'].apply(first_name)
print(df2['freq'].value_counts())
