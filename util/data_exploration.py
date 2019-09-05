# import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns


def print_shape(df):
    print('The shape of the {} is {}'.format(type(df), df.shape))


def print_variable_data_types(df):
    print(df.dtypes)


def check_for_missing_elements(df):
    print(df.isnull().sum())


# Multivariate scatter plots
def plot_distribution(df):
    _, ax = plt.subplots(1, 1, figsize=(14, 6))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.scatterplot(x="SessionId", y=df['ItemId'].astype(int), cmap=cmap, data=df)
    plt.rcParams["ytick.labelsize"] = 7
    plt.show()


# Show value counts for a single variable
def plot_count(df):
    sns.set(style="darkgrid")
    sns.countplot(x=df['ItemId'].astype(int), data=df)
    plt.show()
