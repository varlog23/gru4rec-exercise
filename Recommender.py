import numpy as np
import logging
from gru4rec.gru4rec import GRU4Rec


class Recommender():
    """
    A Recurrent Neural Network based model for Session-based recommendation.
    Based on the following two papers:

    * Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, Hidasi and Karatzoglou, CIKM 2018
    * Personalizing Session-based Recommendation with Hierarchical Recurrent Neural Networks, Quadrana et al, Recsys 2017

    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self,
                 session_layers,
                 user_layers=None,
                 batch_size=32,
                 learning_rate=0.1,
                 momentum=0.0,
                 dropout=None,
                 epochs=10,
                 final_act='linear',
                 personalized=False):
        """
        :param session_layers: number of units per layer used at session level.
            It has to be a list of integers for multi-layer networks, or a integer value for single-layer networks.
        :param user_layers: number of units per layer used at user level. Required only by personalized models.
            It has to be a list of integers for multi-layer networks, or a integer value for single-layer networks.
        :param batch_size: the mini-batch size used in training
        :param learning_rate: the learning rate used in training (Adagrad optimized)
        :param momentum: the momentum coefficient used in training
        :param dropout: dropout coefficients.
            If personalized=False, it's a float value for the hidden-layer(s) dropout.
            If personalized=True, it's a 3-tuple with the values for the dropout of (user hidden, session hidden, user-to-session hidden) layers.
        :param epochs: number of training epochs
        :param personalized: whether to train a personalized model using the HRNN model.
            It will require user ids at prediction time.
        """
        super(Recommender).__init__()
        if isinstance(session_layers, int):
            session_layers = [session_layers]
        if isinstance(user_layers, int):
            user_layers = [user_layers]
        self.session_layers = session_layers
        self.user_layers = user_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        if dropout is None:
            if not personalized:
                dropout = 0.0
            else:
                dropout = (0.0, 0.0, 0.0)
        self.dropout = dropout
        self.epochs = epochs
        self.final_act = final_act
        self.personalized = personalized
        self.pseudo_session_id = 0

    def __str__(self):
        return 'Recommender(' \
               'session_layers={session_layers}, ' \
               'user_layers={user_layers}, ' \
               'batch_size={batch_size}, ' \
               'learning_rate={learning_rate}, ' \
               'momentum={momentum}, ' \
               'dropout={dropout}, ' \
               'epochs={epochs}, ' \
               'final_act={final_act}, ' \
               'personalized={personalized}, ' \
               ')'.format(**self.__dict__)

    def fit(self, train_data):
        if not self.personalized:
            # fit gru4rec
            self.model = GRU4Rec(layers=self.session_layers,
                                 n_epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 learning_rate=self.learning_rate,
                                 momentum=self.momentum,
                                 dropout_p_hidden=self.dropout,
                                 final_act = self.final_act,
                                 session_key='SessionId',
                                 item_key='ItemId',
                                 time_key='Time')

        self.logger.info('Training started')
        self.model.fit(train_data)
        self.logger.info('Training completed')

    def recommend(self, user_profile, user_id=None):
        if not self.personalized:
            for item in user_profile:
                pred = self.model.predict_next_batch(np.array([self.pseudo_session_id]),
                                                     np.array([item]),
                                                     batch=1)
        else:
            if user_id is None:
                raise ValueError('user_id required by personalized models')
            for item in user_profile:
                pred = self.model.predict_next_batch(np.array([self.pseudo_session_id]),
                                                     np.array([item]),
                                                     np.array([user_id]),
                                                     batch=1)
        # sort items by predicted score
        pred.sort_values(0, ascending=False, inplace=True)
        # increase the psuedo-session id so that future call to recommend() won't be connected
        self.pseudo_session_id += 1
        # convert to the required output format
        return [([x.index], x._2) for x in pred.reset_index().itertuples()]
