import pandas as pd
import numpy as np
from preprocess import generate_timestamp

number_of_sequences = 200
min_sequence_length = 1
max_sequence_length = 50
random_seed = 1234


def generate_sequences(sequence_gap, session_id, begin, end, numseqs=number_of_sequences):
    sess = session_id
    all_res = []
    np.random.seed(random_seed)
    for i in range(numseqs):
        rand_sequence_length = np.random.randint(min_sequence_length, max_sequence_length)
        rand_starting_id = np.random.randint(begin, end)
        temp_seq = np.arange(rand_starting_id, rand_starting_id+rand_sequence_length, sequence_gap)
        temp_seq = temp_seq.astype(str)
        df_temp = pd.DataFrame(data=temp_seq, columns=['ItemId'])
        df_temp['Time'] = df_temp.apply(generate_timestamp, axis=1)
        df_temp['SessionId'] = sess
        sess += 1
        all_res.append(df_temp)

    df_res = pd.concat(all_res)
    return df_res


def generate_noise(session_id, begin, end, minseq=min_sequence_length, maxseq=max_sequence_length,
                   numseqs=number_of_sequences):
    sess = session_id
    all_res = []
    np.random.seed(random_seed)
    for i in range(numseqs):
        sequence_length = np.random.randint(minseq, maxseq)
        temp_seq = []
        for j in range(sequence_length):
            temp_seq.append(np.random.randint(begin, end))
        temp_seq = np.asarray(temp_seq)
        temp_seq = temp_seq.astype(str)
        df_temp = pd.DataFrame(data=temp_seq, columns=['ItemId'])
        df_temp['Time'] = df_temp.apply(generate_timestamp, axis=1)
        df_temp['SessionId'] = sess
        sess += 1
        all_res.append(df_temp)

    df_res = pd.concat(all_res)
    return df_res
