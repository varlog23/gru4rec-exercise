import preprocess
from gru4rec.gru4rec import GRU4Rec


def main():
    data = preprocess.get_data()
    print(data)
    gru = GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    gru.fit(data)


if __name__ == "__main__":
    main()
