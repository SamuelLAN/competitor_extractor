import os
import sys

cur_dir = os.path.split(__file__)[0]
root_dir = os.path.split(cur_dir)[0]
sys.path.append(root_dir)

import time
import json
import tensorflow as tf
from load.fixed_mean_sent_emb_loader import Loader
from models.fc import Model
from lib import logs

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

Model.name = 'fixed_mean_sent_emb_fc'
logs.MODEL = Model.name
logs.VARIANT = f'neg_rate_{Model.data_params["neg_rate"]}'


class Train:
    M = Model

    def __init__(self):
        o_loader = Loader(negative_rate=self.M.data_params['neg_rate'], use_cache=True)
        (self.__train_X1, self.__train_X2, self.__train_Y, self.__train_names_1, self.__train_names_2), \
        (self.__val_X1, self.__val_X2, self.__val_Y, self.__val_names_1, self.__val_names_2), \
        (self.__test_X1, self.__test_X2, self.__test_Y, self.__test_names_1, self.__test_names_2) = \
            o_loader.train_val_test(self.M.data_params['train_ratio'], self.M.data_params['val_ratio'])

        logs.new_paragraph(True)
        logs.add(self.M.name, 'data_shape', json.dumps({
            'train_x': self.__train_X1.shape,
            'train_y': self.__train_Y.shape,
            'val_x': self.__val_X1.shape,
            'val_y': self.__val_Y.shape,
            'test_x': self.__test_X1.shape,
            'test_y': self.__test_Y.shape,
        }), logs.LEVEL_DATA, True)

    def train(self):
        print('\nBuilding model ({}) ...'.format(self.M.TIME))
        self.model = self.M()

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(
            (self.__train_X1, self.__train_X2),
            self.__train_Y,
            (self.__val_X1, self.__val_X2),
            self.__val_Y
        )
        train_time = time.time() - start_time
        print('\nFinish training')

        logs.add(self.M.name, 'training_time', f'{train_time}')

    def test(self, load_model=False):
        # load the model
        if load_model:
            self.model = self.M(finish_train=True)
            self.model.train((self.__train_X1, self.__train_X2), self.__train_Y)
            self.__train_time = 0.

        self.evaluate('training', self.__train_X1, self.__train_X2, self.__train_Y)
        self.evaluate('val', self.__val_X1, self.__val_X2, self.__val_Y)
        self.evaluate('test', self.__test_X1, self.__test_X2, self.__test_Y, True, self.__test_names_1,
                      self.__test_names_2)

    def evaluate(self, prefix, X1, X2, Y, record_details=False, names_1=None, names_2=None):
        ret = self.model.test((X1, X2), Y)

        logs.new_line(True)
        for indicator, score in ret.items():
            logs.add(self.M.name, f'{prefix}_evaluation', f'{indicator}: {score}', logs.LEVEL_RET, True)

        if record_details:
            predict_y = self.model.predict_label((self.__test_X1, self.__test_X2))

            logs.new_line(True)
            for i, v in enumerate(predict_y):
                logs.add(
                    self.M.name,
                    'test_samples',
                    json.dumps({
                        'ret': "success" if v == Y[i] else "fail",
                        'predict': int(v),
                        'ground_truth': int(Y[i]),
                        'name_1': names_1[i],
                        'name_2': names_2[i],
                    }),
                    logs.LEVEL_DETAIL,
                    True
                )


o_train = Train()
o_train.train()
o_train.test(False)
