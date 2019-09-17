import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv('C:/Users/User/Desktop/ml_course/datasets/telecom_churn.csv')
print(df.head())
df['Total day minutes'].hist()
plt.show()
sns.boxplot(df['Total day minutes'])
plt.show()
sns.countplot(df['State'])
plt.show()
feat = [f for f in df.columns if 'charge' in f]
sns.pairplot(df[feat])
plt.show()
plt.scatter(df[df['Churn']]['Total eve charge'], df[df['Churn']]['Total intl charge'], color='blue',
            label='churn')
plt.scatter(df[~df['Churn']]['Total eve charge'], df[~df['Churn']]['Total intl charge'], color='orange',
            label='loyal')
plt.xlabel('Вечерние начисления')
plt.ylabel('Межнар. начисления')
plt.title('классный график')
plt.legend()

sns.pairplot(df)
plt.show()

