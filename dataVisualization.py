import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

#Loading Boston House Price DataSet
boston_dataset = load_boston()

#Loading the dataset into Panda dataframe
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

#Tabulazing the dataset in table format
df['PRICE'] = boston_dataset.target

#Plotting the data for AGE Vs PRICE in bar type
df.plot(kind='bar',x='AGE',y='PRICE',title='AGE VS PRICE')

#Displaying the plotted visual
plt.show()