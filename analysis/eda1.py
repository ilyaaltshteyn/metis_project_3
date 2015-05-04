import pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_cat']
file = '/Users/ilya/metis/week4/metis_project_3/analysis/census_income_data.csv'
data = pd.read_csv(file, names = cols)

data = data.drop('fnlwgt', axis = 1)
data = data.drop('education', axis = 1)
data = data.drop('capital_gain', axis = 1)
data = data.drop('capital_loss', axis = 1)

print data.describe()
data = data[data.age >= 18]

sns.barplot(data.income_cat)
plt.show()
data = data[data.income_cat != 14]

sns.barplot(data.sex)
plt.show()

# Print out unique values of categorical columns to look for nonsense:
print data.race.unique()
print data.workclass.unique()
print data.marital_status.unique()
print data.occupation.unique()

data.to_csv('/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv')

