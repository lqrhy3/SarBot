from math import log2
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Создание датафрейма с dummy variables
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
    out.drop(feature_list, axis=1, inplace=True)
    return out


# Некоторые значения признаков есть в тесте, но нет в трейне и наоборот
def intersect_features(train, test):
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


features = ['Внешность', 'Алкоголь_в_напитке',
            'Уровень_красноречия', 'Потраченные_деньги']
df_train = {}
df_train['Внешность'] = ['приятная', 'приятная', 'приятная', 'отталкивающая',
                         'отталкивающая', 'отталкивающая', 'приятная']
df_train['Алкоголь_в_напитке'] = ['да', 'да', 'нет', 'нет', 'да', 'да', 'да']
df_train['Уровень_красноречия'] = ['высокий', 'низкий', 'средний', 'средний', 'низкий',
                                   'высокий', 'средний']
df_train['Потраченные_деньги'] = ['много', 'мало', 'много', 'мало', 'много',
                                  'много', 'много']
df_train['Поедет'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])

df_train = create_df(df_train, features)
df_test = {}
df_test['Внешность'] = ['приятная', 'приятная', 'отталкивающая']
df_test['Алкоголь_в_напитке'] = ['нет', 'да', 'да']
df_test['Уровень_красноречия'] = ['средний', 'высокий', 'средний']
df_test['Потраченные_деньги'] = ['много', 'мало', 'много']
df_test = create_df(df_test, features)
y = df_train['Поедет']
df_train, df_test = intersect_features(train=df_train, test=df_test)

print(df_test.head(), df_train.head())
print(y.value_counts())
entropy = -(4/7 * log2(4/7) + 3/7 * log2(3/7))
print(entropy)
print(df_train.columns)
df_train = pd.concat([df_train, y], axis=1)
df_train.sort_values(by='Внешность_приятная', inplace=True)
counter = 0
for index in df_train['Внешность_приятная']:
    if index == 1:
        break
    else:
        counter += 1
first = df_train.iloc[0:counter, :]
second = df_train.iloc[counter:, :]
print(first['Внешность_приятная'].nunique(), second['Внешность_приятная'].nunique(), sep=',')
print(first['Поедет'].value_counts(), second['Поедет'].value_counts(), sep='\n\n')
entropy1 = -(2/3 * log2(2/3) + 1/3 * log2(1/3))
entropy2 = -(3/4 * log2(3/4) + 1/4 * log2(1/4))
print(entropy1, entropy2)
print(entropy - (3/7)*entropy1 - (4/7)*entropy2)


def entropy_calc(a_list):
    uniq = set(a_list)
    n = len(uniq)
    uniqe = list(uniq)
    entr = 0
    for i in range(n):
        entr += (a_list.count(uniqe[i])/len(a_list)) * log2(a_list.count(uniqe[i])/len(a_list))
    return -entr


balls_left = [1 for k in range(8)] + [0 for j in range(5)]
print(entropy_calc(balls_left))
print(entropy_calc([1, 2, 3, 4, 5, 6]))



