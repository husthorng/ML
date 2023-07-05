# load the csv data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
url ="https://raw.githubusercontent.com/husthorng/ML/main/Iris.csv"
df = pd.read_csv(url)

df = df.drop(columns = ['Id'])
print(df.head())
print(df.describe())
# to get basic info about datatypes
print(df.info())
# to display no. of samples on each class
print(df['Species'].value_counts())
# check for null values,
# If any NULL values are present, we have to fill all the NULL values before proceeding to model training.
print(df.isnull().sum())
#to find some patterns (or) relations within the data
# histograms
#his=df['SepalLengthCm'].hist()
#plt.hist(df['SepalLengthCm'],edgecolor='black',align='right')
#plt.hist(df['SepalWidthCm'],edgecolor='black',align='right')
#plt.hist(df['PetalLengthCm'],edgecolor='black',align='right')
#plt.hist(df['Species'],edgecolor='black',align='right')
#plt.show()
# create list of colors and class labels
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()