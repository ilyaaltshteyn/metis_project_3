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
plt.title('Breakdown of income categories in dataset')
plt.show()
data = data[data.income_cat != 14]

sns.barplot(data.sex)
plt.title('Breakdown of sex categories in dataset')
plt.show()

# Print out unique values of categorical columns to look for nonsense:
print data.race.unique()
print data.workclass.unique() # Bin 'without-pay' and 'never-worked'
data['workclass'] = data.workclass.replace([' Without-pay', ' Never-worked'], 'No-pay/never-worked')
print data.marital_status.unique()
print data.relationship.unique()
print data.occupation.unique()
print data.native_country.unique() #Convert to USA yes/no
data['native_country'] = [1 if x == ' United-States' else 0 for x in data.native_country]
print data.sex.unique()
data['sex'] = data.sex.replace([' Male', ' Female'], [1,0])


print data.describe()
data.to_csv('/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv',
    index = False, header = True)

