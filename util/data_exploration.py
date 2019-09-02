# For data wrangling
import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Other imports
import preprocess


def print_shape(df):
    print('The shape of the {} is {}'.format(type(df), df.shape))


def print_variable_data_types(df):
    print(df.dtypes)


def check_for_missing_elements(df):
    print(df.isnull().sum())


def print_unique_count(df):
    print(df.isnull().sum())


# Multivariate scatter plots
def plot_distribution(df):
    _, ax = plt.subplots(1, 1, figsize=(14, 6))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.scatterplot(x="SessionId", y="ItemId", cmap=cmap, data=df)
    plt.show()


# Show value counts for a single variable
def plot_count(df):
    sns.set(style="darkgrid")
    sns.countplot(x="ItemId", data=df)
    plt.show()



