import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


def get_data(filepath):
    current_path = os.getcwd()
    df = pd.read_csv(current_path + filepath)
    return df


def get_missing_dates(df):
    # Finding holes in the dates with some missing values
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    start_date = df.iloc[0].loc["date"]
    end_date = df.iloc[-1].loc["date"]
    date_range = pd.date_range(start_date, end_date)
    df.reindex(date_range).isnull().all(1)
    diff_range = date_range.difference(df["date"])
    return diff_range


def insert_missing_dates(df):
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    start_date = df.iloc[0].loc["date"]
    end_date = df.iloc[-1].loc["date"]
    date_range = pd.date_range(start_date, end_date)
    df.set_index("date").reindex(date_range)
    print(get_missing_dates(df))
    return df

'''
def prepare_raw_data(datapath, recoveries_file, raw_file, refined_file, country):
    interesting_columns = [
        "iso_code",
        "date",
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
        "total_tests",
        "new_tests",
        "positive_rate",
        "population",
    ]

    df_recoveries = get_data(datapath + recoveries_file)
    missing_recovery_dates = get_missing_dates(df_recoveries)
    print(missing_recovery_dates)

    # Saving italian rows without null
    df = get_data(datapath + raw_file)
    df_ita = df[df["location"] == country]
    df_ita_filtered = df_ita[interesting_columns].dropna()
    df_ita_filtered.to_csv(os.getcwd() + datapath + refined_file)
    missing_ita_dates = get_missing_dates(df_ita_filtered)
    print(missing_ita_dates)

    # Missing code for merging, done easily with excel
    return
'''

def main():
    week_len = 7
    # PARAMS
    country = "Italy"
    datapath = "/data"
    raw_file = "/raw.csv"
    processed_file = "/processed.csv"

    #FIX with real value
    population_it = 50000000

    df = get_data(datapath + raw_file)

    # new_df = (
    #     df[
    #         [
    #             "totale_positivi",
    #             "variazione_totale_positivi",
    #             "nuovi_positivi",
    #             "dimessi_guariti",
    #             "deceduti",
    #             "totale_ospedalizzati",
    #             "isolamento_domiciliare",
    #         ]
    #     ]
    #     .groupby(np.arange(len(df)) // week_len)
    #     .mean()
    # )
    # new_df["nuovi_positivi"] = new_df["nuovi_positivi"].apply(lambda x: x * week_len)
    # new_df["dimessi_guariti"] = new_df["dimessi_guariti"].apply(lambda x: x * week_len)
    # new_df["deceduti"] = new_df["deceduti"].apply(lambda x: x * week_len)

    # new_df['totale_positivi'] = new_df['totale_positivi'].apply(np.floor)
    # new_df['variazione_totale_positivi'] = new_df['variazione_totale_positivi'].apply(np.floor)
    # new_df['totale_ospedalizzati'] = new_df['totale_ospedalizzati'].apply(np.floor)
    # new_df['isolamento_domiciliare'] = new_df['isolamento_domiciliare'].apply(np.floor)
    # df = new_df
    # df.to_csv(os.getcwd() + datapath + processed_file)
    df["data"] = pd.to_datetime(df["data"])
    df.set_index("data", inplace=True)
    total_positives = df["totale_positivi"].resample('W').last()
    recovered = df["dimessi_guariti"].resample('W').last()
    deaths = df["deceduti"].resample('W').last()

    # replace date with 0 indexed integer
    new_df = pd.DataFrame(
        {
            "totale_positivi": total_positives,
            "dimessi_guariti": recovered,
            "deceduti": deaths,
        },
    )



    new_df.set_index(np.arange(len(new_df)), inplace=True)

    new_df.to_csv(os.getcwd() + datapath + processed_file)


if __name__ == "__main__":
    main()
