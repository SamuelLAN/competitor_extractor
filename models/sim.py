import os
import time
import json
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from lib import path_lib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from lib import logs

keras = tf.keras
tfv1 = tf.compat.v1


class Model:
    name = 'Similarity'

    data_params = {
        'neg_rate': 1,
        'neg_rate_train': 1,
        'neg_rate_val': 2,
        'neg_rate_test': 3,
        'train_ratio': 0.81,
        'val_ratio': 0.09,
        'test_ratio': 0.1,
    }

    model_params = {
        'top_k': 50,
        'threshold': 0.1,
    }

    evaluate_label_dict = {
        'acc': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }

    evaluate_score_dict = {
        'auc': roc_auc_score,
    }

    TIME = str(time.strftime('%Y_%m_%d_%H_%M_%S'))

    def __init__(self):
        # create directories for tensorboard files and model files
        self.create_dir()
        self.__log()

    def __log(self):
        logs.new_line()
        logs.add(self.name, 'data_params', json.dumps(self.data_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'model_params', json.dumps(self.model_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'model_dir', self.model_dir, logs.LEVEL_PATH, True)

    def create_dir(self):
        # create model path
        self.model_dir = path_lib.create_dir_in_root('runtime', 'models', self.name)

    def train(self, X, names, use_cache=True):
        result_path = os.path.join(
            self.model_dir,
            f'd_name_2_similar_names_top_{self.model_params["top_k"]}_threshold_{self.model_params["threshold"]}.json')

        if use_cache and os.path.exists(result_path):
            self.d_name_2_similar_names = path_lib.read_cache(result_path)
            return

        # calculate the cosine distance
        distances = cdist(X, X, 'cosine')

        # get the results with the top k minimal cosine distance
        for i in range(len(distances)):
            distances[i, i] = 2
        # distances[distances < 1e-15] = 2
        similarities = 1 - np.tanh(distances)
        top_k_idx = similarities.argsort()[:, -self.model_params['top_k']:]

        # save results
        self.d_name_2_similar_names = {}
        for i, name in enumerate(names):
            tmp_idx = top_k_idx[i][top_k_idx[i] > self.model_params['threshold']]
            self.d_name_2_similar_names[name] = names[tmp_idx]

        path_lib.cache(result_path, self.d_name_2_similar_names)

    def predict_one_label(self, name_pair):
        name_1, name_2 = name_pair
        if name_1 not in self.d_name_2_similar_names or name_2 not in self.d_name_2_similar_names:
            return

        return max(
            int(name_2 in self.d_name_2_similar_names[name_1]),
            int(name_1 in self.d_name_2_similar_names[name_2])
        )

    def predict_labels(self, names_1, names_2):
        names = list(zip(names_1, names_2))
        return np.array(list(map(self.predict_one_label, names)), np.int32)

    def test(self, y, names_1, names_2):
        """ Evaluate the performance """
        ret = {}

        # pred_scores = self.predict_score(x)
        # pred_labels = np.argmax(pred_scores, axis=-1)
        pred_labels = self.predict_labels(names_1, names_2)

        for indicator, func in self.evaluate_label_dict.items():
            ret[indicator] = func(y, pred_labels)

        # for indicator, func in self.evaluate_score_dict.items():
        #     ret[indicator] = func(y, pred_scores[:, 1])

        return ret
