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


csv_files = get_all_files(file_path)
df_from_each_file = []
for index, f in enumerate(csv_files):
    # Select required columns from the CSV files
    df_temp = pd.read_csv(f, usecols=["href", "name", "identifiyng type", "type", "required"])
    # Assign a session ID for every csv file
    df_temp['session_id'] = index
    df_from_each_file.append(df_temp)

# Concatenate DataFrame's
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
# Generate fictitious timestamp in the order
concatenated_df['ts'] = concatenated_df.apply(generate_timestamp, axis=1)
