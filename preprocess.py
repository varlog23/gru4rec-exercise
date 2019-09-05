import pandas as pd
import glob
import os
import time
import numpy as np


def generate_timestamp(row):
    current_time = int(time.time())
    return current_time + int(row.name)


def get_all_files(path, extension="*.csv"):
    return glob.glob(os.path.join(path, extension))


def print_all_columns(df):
    for col in df.columns:
        print(col)


def get_data(file_path):
    csv_files = get_all_files(file_path)
    df_from_each_file = []
    for index, f in enumerate(csv_files):
        # Select required columns from the CSV files
        df_temp = pd.read_csv(f, usecols=["href", "name", "identifiyng type", "type", "required", "action"])
        # Assign a session ID for every csv file
        df_temp['SessionId'] = index
        df_from_each_file.append(df_temp)

    # Concatenate DataFrame's from every test case
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    # Generate fictitious timestamp because the selenium IDE does not record this and the model needs this feature
    concatenated_df['Time'] = concatenated_df.apply(generate_timestamp, axis=1)
    # Create a column that will hold a unique identity for the "item" (GUI component)
    concatenated_df["component"] = np.nan

    # Handle submit type when name column is empty
    concatenated_df.loc[concatenated_df['component'].isnull() & concatenated_df['name'].isnull() & (
            concatenated_df['type'] == 'submit'), 'component'] = \
        concatenated_df['type']

    # Otherwise name (not null) becomes the component
    concatenated_df.loc[concatenated_df['component'].isnull() & concatenated_df['name'].notnull(), 'component'] = \
        concatenated_df['name']

    # When name is null and href isn't null, href is the component
    concatenated_df.loc[concatenated_df['component'].isnull() & concatenated_df['name'].isnull() & concatenated_df[
        'href'].notnull(), 'component'] = \
        concatenated_df['href']

    # Drop rows without component
    concatenated_df = concatenated_df.dropna(subset=['component'])

    # Assign unique ID to every "action + component"
    concatenated_df['ItemId'] = concatenated_df.groupby(['action', 'component'], sort=False).ngroup()

    # concatenated_df = concatenated_df.assign(
    #     ItemId=concatenated_df.index.to_series().groupby(
    #         [concatenated_df.action,concatenated_df.component], sort=False
    #     ).ngroup().map('{}'.format)
    # )[['ItemId'] + concatenated_df.columns.tolist()]

    # Select necessary columns required by the model
    data = concatenated_df[['SessionId', 'ItemId', 'Time']].reset_index(drop=True)
    dict_of_itemid = concatenated_df.drop_duplicates(subset='ItemId', keep="first")[['ItemId', 'component', 'action']]
    data['ItemId'] = data['ItemId'].apply(str)
    return data, dict_of_itemid

