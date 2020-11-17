import os
import sys

cur_dir = os.path.split(__file__)[0]
root_dir = os.path.split(cur_dir)[0]
sys.path.append(root_dir)

import time
import json
from load.fixed_mean_sent_emb_for_a_company_loader import Loader as CLoader
from load.fixed_mean_sent_emb_loader_v2 import Loader
from models.sim import Model
from lib import logs

Model.name = 'fixed_mean_sent_emb_similarity'
logs.MODEL = Model.name


class Train:
    M = Model

    def __init__(self):
        company_loader = CLoader(use_cache=True)
        train_loader = Loader(self.M.data_params['neg_rate_train'],
                              0, self.M.data_params['train_ratio'], use_cache=False)
        val_loader = Loader(self.M.data_params['neg_rate_val'],
                            self.M.data_params['train_ratio'],
                            self.M.data_params['train_ratio'] + self.M.data_params['val_ratio'], use_cache=False)
        test_loader = Loader(self.M.data_params['neg_rate_test'],
                             self.M.data_params['train_ratio'] + self.M.data_params['val_ratio'], 1.0, use_cache=False)

        self.__X, self.__names = company_loader.all()
        self.__train_X1, self.__train_X2, self.__train_Y, self.__train_names_1, self.__train_names_2 = train_loader.all()
        self.__val_X1, self.__val_X2, self.__val_Y, self.__val_names_1, self.__val_names_2 = val_loader.all()
        self.__test_X1, self.__test_X2, self.__test_Y, self.__test_names_1, self.__test_names_2 = test_loader.all()

        logs.new_paragraph(True)
        logs.add(self.M.name, 'data_shape', json.dumps({
            'X': self.__X.shape,
            'train_x': self.__train_X1.shape,
            'train_y': self.__train_Y.shape,
            'val_x': self.__val_X1.shape,
            'val_y': self.__val_Y.shape,
            'test_x': self.__test_X1.shape,
            'test_y': self.__test_Y.shape,
        }), logs.LEVEL_DATA, True)

    def train(self, use_cache=True):
        print('\nBuilding model ({}) ...'.format(self.M.TIME))
        self.model = self.M()

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(self.__X, self.__names, use_cache)
        train_time = time.time() - start_time
        print('\nFinish training')

        logs.add(self.M.name, 'training_time', f'{train_time}')

    def test(self, load_model=False):
        # load the model
        if load_model:
            self.model = self.M()
            self.model.train(self.__X, self.__names)
            self.__train_time = 0.

        self.evaluate('training', self.__train_X1, self.__train_X2, self.__train_Y,
                      False, self.__train_names_1, self.__train_names_2)
        self.evaluate('val', self.__val_X1, self.__val_X2, self.__val_Y,
                      False, self.__val_names_1, self.__val_names_2)
        self.evaluate('test', self.__test_X1, self.__test_X2, self.__test_Y,
                      False, self.__test_names_1, self.__test_names_2)

    def evaluate(self, prefix, X1, X2, Y, record_details=False, names_1=None, names_2=None):
        ret = self.model.test(Y, names_1, names_2)

        logs.new_line(True)
        for indicator, score in ret.items():
            logs.add(self.M.name, f'{prefix}_evaluation', f'{indicator}: {score}', logs.LEVEL_RET, True)

        if record_details:
            predict_y = self.model.predict_labels(names_1, names_2)

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
                    False
                )


# for i, top_k in enumerate(list(range(370, 510, 10))):
Model.model_params['top_k'] = 410
logs.VARIANT = f'top_{Model.model_params["top_k"]}_threshold_{Model.model_params["threshold"]}_v5'

o_train = Train()
o_train.train(use_cache=False)
o_train.test(False)
