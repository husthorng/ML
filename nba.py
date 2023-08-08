#nba.py
# importing pandas package
import pandas as pd
 
# making data frame from csv file
url ='https://raw.githubusercontent.com/husthorng/ML/main/nba.csv'
data = pd.read_csv(url,index_col ="Name")
 
# retrieving row by loc method
first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]
 
 
print(first, "\n\n\n", second)