import numpy as np
from collections import Counter


def create_sequence_for_sessions(df):
    # group by session id and concat item_id
    groups = df.groupby('SessionId')
    # convert item ids to string, then aggregate them to lists
    aggregated = groups['ItemId'].apply(lambda x: list(map(str, x)))
    aggregated = aggregated.to_frame(name='sequence')

    init_ts = groups['Time'].min()

    result = aggregated.join(init_ts)
    result.reset_index(inplace=True)
    return result


def print_dataset_info(df):
    seq_df = create_sequence_for_sessions(df)
    cnt = Counter()
    seq_df.sequence.map(cnt.update)
    sequence_length = seq_df.sequence.map(len).values
    print('Number of items: {}'.format(len(cnt)))
    print('Number of sessions: {}'.format(len(seq_df)))

    print('\nSession length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
        sequence_length.mean(),
        np.quantile(sequence_length, 0.5),
        sequence_length.min(),
        sequence_length.max()))

    print('Top 5 most common items as (item,count): {}'.format(cnt.most_common(5)))
