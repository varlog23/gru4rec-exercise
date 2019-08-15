import pandas as pd
import glob
import os
import time
import numpy as np
import random

file_path = r'/home/varun/Downloads/Dataset/'


def generate_timestamp(row):
    current_time = int(time.time())
    return current_time + int(row.name)


def get_all_files(path, extension="*.csv"):
    return glob.glob(os.path.join(path, extension))


def print_all_columns(df):
    for col in df.columns:
        print(col)


def get_data():
    csv_files = get_all_files(file_path)
    df_from_each_file = []
    for index, f in enumerate(csv_files):
        # Select required columns from the CSV files
        df_temp = pd.read_csv(f, usecols=["href", "name", "identifiyng type", "type", "required"])
        # Assign a session ID for every csv file
        df_temp['SessionId'] = index
        df_from_each_file.append(df_temp)

    # Concatenate DataFrame's
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    # Generate fictitious timestamp in the order
    concatenated_df['Time'] = concatenated_df.apply(generate_timestamp, axis=1)
    # Create a column that will hold a unique identity for the component
    concatenated_df["reference"] = np.nan

    concatenated_df.loc[concatenated_df['reference'].isnull() & concatenated_df['name'].isnull() & (
            concatenated_df['type'] == 'submit'), 'reference'] = \
        concatenated_df['type']

    concatenated_df.loc[concatenated_df['reference'].isnull() & concatenated_df['name'].notnull(), 'reference'] = \
        concatenated_df['name']

    concatenated_df.loc[concatenated_df['reference'].isnull() & concatenated_df['name'].isnull() & concatenated_df[
        'href'].notnull(), 'reference'] = \
        concatenated_df['href']

    # Assign unique ID to every "reference"
    concatenated_df['ItemId'] = concatenated_df.groupby(['reference'], sort=False).ngroup()
    # Drop rows without reference
    concatenated_df = concatenated_df.dropna(subset=['reference'])
    # Select necessary columns required by the model
    data = concatenated_df[['SessionId','ItemId', 'Time']]
    return data

