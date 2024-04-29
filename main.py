import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os 

#PARAMS
country = 'Italy'
interesting_columns = ['iso_code', 'date','total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_tests', 'new_tests', 'positive_rate', 'population']

datapath = '/data'
raw_file = '/raw.csv'
refined_file = '/refined.csv'


def get_data(filepath):
    current_path = os.getcwd()
    df = pd.read_csv(current_path+filepath)
    return df


df = get_data(datapath+raw_file)

#Saving italian rows without null
df_ita = df[df['location'] == country]
df_ita_filtered = df_ita[interesting_columns].dropna()
df_ita_filtered.to_csv(os.getcwd()+datapath+refined_file)


#Finding holes in the dates with some missing values
df_ita_filtered['date'] = pd.to_datetime(df_ita_filtered['date'])
start_date = df_ita_filtered.iloc[0].loc['date']
end_date = df_ita_filtered.iloc[-1].loc['date']


date_range = pd.date_range(start_date, end_date)
df_ita_filtered.reindex(date_range
                        ).isnull().all(1)
diff_range = date_range.difference(df_ita_filtered['date'])
print(diff_range)

#Further investigation on missing values for those rows 
