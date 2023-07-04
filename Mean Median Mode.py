import pandas as pd
import numpy
import statistics
#url = "https://raw.githubusercontent.com/husthorng/hello-world/horng_1/110hiwinstud.csv"
url ="https://raw.githubusercontent.com/husthorng/ML/main/CAR.csv"
name_records = pd.read_csv(url,encoding='utf-8')
#print(name_records.head())
speed=name_records["Speed"]
x = numpy.mean(speed)
y = numpy.median(speed)
z = statistics.mode(speed)
print("\n mean=",x,"\n","median=",y,"\n","mode=",z,"\n",)
