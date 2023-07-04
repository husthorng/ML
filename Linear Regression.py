import pandas as pd
from scipy import stats
import numpy
import statistics
#url = "https://raw.githubusercontent.com/husthorng/hello-world/horng_1/110hiwinstud.csv"
url ="https://raw.githubusercontent.com/husthorng/ML/main/CAR.csv"
name_records = pd.read_csv(url,encoding='utf-8')
#print(name_records.head())
#speed=name_records["Speed"]
#x = numpy.mean(speed)
#y = numpy.median(speed)
#z = statistics.mode(speed)
#print("\n mean=",x,"\n","median=",y,"\n","mode=",z,"\n",)
x=name_records["Age"]
y=name_records["Speed"]
import matplotlib.pyplot as plt
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
