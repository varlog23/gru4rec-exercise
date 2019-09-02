import pandas as pd
import numpy as np
from preprocess import generate_timestamp

number_of_sequences = 100


def generate_sequences(sequence_gap, session_id, begin, end):
    # session_id = 40
    sess = session_id
    all_res = []
    for i in range(number_of_sequences):
        sequence_length = np.random.randint(3, 25)
        # sequence_gap = np.random.randint(1, 3)
        starting_id = np.random.randint(begin, end)
        temp_seq = np.arange(starting_id, starting_id+sequence_length, sequence_gap)
        temp_seq = temp_seq.astype(str)
        df_temp = pd.DataFrame(data=temp_seq, columns=['ItemId'])
        df_temp['Time'] = df_temp.apply(generate_timestamp, axis=1)
        df_temp['SessionId'] = sess
        sess += 1
        all_res.append(df_temp)

    df_res = pd.concat(all_res)
    return df_res
