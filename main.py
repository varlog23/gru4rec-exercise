import preprocess
import util.data_exploration
import util.data_utils
import util.gendata
from Recommender import Recommender
from util.split import test_train_split

train_file_path = r'/home/varun/Downloads/Dataset/TrainAction/'


def generate_synthetic_data(maxid=0):
    synthetic_data = util.gendata.generate_sequences(1, maxid+1, 50, 150)

    temp_df = util.gendata.generate_sequences(2, synthetic_data['SessionId'].max()+1, 160, 300)
    synthetic_data = synthetic_data.append(temp_df)

    temp_df = util.gendata.generate_sequences(1, synthetic_data['SessionId'].max()+1, 1, 500)
    synthetic_data = synthetic_data.append(temp_df)

    temp_df = util.gendata.generate_sequences(4, synthetic_data['SessionId'].max()+1, 400, 500)
    synthetic_data = synthetic_data.append(temp_df)

    temp_df = generate_noise(synthetic_data['SessionId'].max())
    synthetic_data = synthetic_data.append(temp_df)

    return synthetic_data


def generate_noise(maxid=0):
    noisy_data = util.gendata.generate_noise(maxid+1, 1, 500, minseq=1, maxseq=50, numseqs=150)

    temp_df = util.gendata.generate_noise(noisy_data['SessionId'].max() + 1, 1, 500, minseq=1, maxseq=15, numseqs=50)
    noisy_data = noisy_data.append(temp_df)

    noisy_data_as_sessions = util.data_utils.create_sequence_for_sessions(noisy_data)
    print(noisy_data_as_sessions)
    # Plot noise
    # util.data_exploration.plot_distribution(noisy_data)
    # util.data_exploration.plot_count(noisy_data)
    return noisy_data


if __name__ == "__main__":
    # Import data
    data, dict_of_items = preprocess.get_data(train_file_path)

    # Append Synthetic data(contains noise)
    data = data.append(generate_synthetic_data(data['SessionId'].max()))

    # Split dataset
    data, valid = test_train_split(data, 1.0)
    util.data_utils.print_dataset_info(data)
    data_as_sessions = util.data_utils.create_sequence_for_sessions(data)
    print(data_as_sessions)

    # Plot data to visualize
    util.data_exploration.plot_distribution(data)
    util.data_exploration.plot_count(data)

    # Instantiate the model
    recommender = Recommender(session_layers=[100],
                              batch_size=2,
                              learning_rate=0.01,
                              momentum=0.1,
                              dropout=0.1,
                              epochs=2,
                              personalized=False)
    recommender.fit(data)

    # Test manually for prediction
    item_predicted = recommender.recommend(['1'])
    print('Prediction for [1] is: {}'.format(item_predicted))

    item_predicted = recommender.recommend(['8', '9'])
    print('Prediction for [8,9] is: {}'.format(item_predicted))

    item_predicted = recommender.recommend(['1', '2'])
    print('Prediction for [1,2] is: {}'.format(item_predicted))

    item_predicted = recommender.recommend(['1', '2', '3'])
    print('Prediction for [1,2,3] is: {}'.format(item_predicted))
