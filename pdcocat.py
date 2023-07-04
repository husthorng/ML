#https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC2-1%E8%AC%9B-%E5%A6%82%E4%BD%95%E7%8D%B2%E5%8F%96%E8%B3%87%E6%96%99-sklearn%E5%85%A7%E5%BB%BA%E8%B3%87%E6%96%99%E9%9B%86-baa8f027ed7b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import sklearn
from sklearn.datasets import load_iris
iris=load_iris()
print(iris.keys())
#print(iris['DESCR'])
#print(iris['data'])
print(iris['feature_names'])
x=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target_names'])
data=pd.concat([x,y],axis=1)
print(data)