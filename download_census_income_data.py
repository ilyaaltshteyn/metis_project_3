import urllib2, pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

data = pd.read_csv(url, header = -1)

data.to_csv('census_income_data.csv')