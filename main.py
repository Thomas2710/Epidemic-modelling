import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os 

#PARAMS
country = 'Italy'
interesting_columns = ['iso_code', 'date','total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_tests', 'new_tests', 'positive_rate', 'population']

datapath = '/data'
raw_file = '/raw.csv'
recoveries_file = '/recovery.csv'
refined_file = '/refined.csv'


def get_data(filepath):
    current_path = os.getcwd()
    df = pd.read_csv(current_path+filepath)
    return df

def get_missing_dates(df):
    #Finding holes in the dates with some missing values
    df['date'] = pd.to_datetime(df['date'], format = 'mixed')
    start_date = df.iloc[0].loc['date']
    end_date = df.iloc[-1].loc['date']
    date_range = pd.date_range(start_date, end_date)
    df.reindex(date_range).isnull().all(1)
    diff_range = date_range.difference(df['date'])
    return diff_range

def insert_missing_dates(df):
    df['date'] = pd.to_datetime(df['date'], format = 'mixed')
    start_date = df.iloc[0].loc['date']
    end_date = df.iloc[-1].loc['date']
    date_range = pd.date_range(start_date, end_date)
    df.set_index('date').reindex(date_range)
    print(get_missing_dates(df))
    return df

df = get_data(datapath+raw_file)
df_recoveries = get_data(datapath+recoveries_file)
missing_recovery_dates = get_missing_dates(df_recoveries)
print(missing_recovery_dates)


#Saving italian rows without null
df_ita = df[df['location'] == country]
df_ita_filtered = df_ita[interesting_columns].dropna()
df_ita_filtered.to_csv(os.getcwd()+datapath+refined_file)
missing_ita_dates = get_missing_dates(df_ita_filtered)
print(missing_ita_dates)

#Further investigation on missing values for those rows 
