import os
import time
import json
import numpy as np
import tensorflow as tf
from lib.tf_models.fc import FC
from lib.tf_metrics import metrics
from lib.tf_callback.board import Board
from lib.tf_callback.saver import Saver
from lib import path_lib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from lib import logs

keras = tf.keras
tfv1 = tf.compat.v1


class Model:
    name = 'FC'

    data_params = {
        'neg_rate': 1,
        'neg_rate_train': 1,
        'neg_rate_val': 2,
        'neg_rate_test': 3,
        'train_ratio': 0.81,
        'val_ratio': 0.09,
        'test_ratio': 0.1,
    }

    train_params = {
        'learning_rate': 1e-6,
        'batch_size': 64,
        'epoch': 1000,
        'early_stop': 30,
    }

    model_params = {
        # 'hidden_layers': [],
        'hidden_layers': [1024, 256],
    }

    tb_params = {
        'histogram_freq': 0,
        'update_freq': 'epoch',
        'write_grads': False,
        'write_graph': True,
        'write_images': False,
        'profile_batch': 0,
    }

    compile_params = {
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
        'customize_loss': True,
        'label_smooth': 0.1,
        'metrics': [metrics.tf_accuracy, metrics.tf_precision, metrics.tf_recall, metrics.tf_f1],
    }

    monitor_params = {
        'monitor': 'val_tf_f1',
        'mode': 'max',  # for the "name" monitor, the "min" is best;
        'early_stop': train_params['early_stop'],
        'start_train_monitor': 'tf_accuracy',
        'start_train_monitor_value': 0.5,
        'start_train_monitor_mode': 'max',
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['monitor']
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

    def __init__(self, finish_train=False):
        self.__finish_train = finish_train

        # create directories for tensorboard files and model files
        self.create_dir()

        # build models
        self.build()

        # for using model.fit, set callbacks for the training process
        self.set_callbacks()

        self.__log()

    def __log(self):
        logs.new_line()
        logs.add(self.name, 'data_params', json.dumps(self.data_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'train_params', json.dumps(self.train_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'model_params', json.dumps(self.model_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'monitor_params', json.dumps(self.monitor_params), logs.LEVEL_PARAM, True)
        logs.add(self.name, 'model_dir', self.model_dir, logs.LEVEL_PATH, True)
        logs.add(self.name, 'tensorboard_dir', self.tb_dir, logs.LEVEL_PATH, True)

    def create_dir(self):
        # create tensorboard path
        self.tb_dir = path_lib.create_dir_in_root('runtime', 'tensorboard', self.name, self.TIME)

        # create model path
        self.model_dir = path_lib.create_dir_in_root('runtime', 'models', self.name, self.TIME)
        self.checkpoint_path = os.path.join(self.model_dir, self.name + self.checkpoint_params['extend_name'])

    def build(self):
        self.model = FC(self.model_params['hidden_layers'])

    def loss(self, y_true, y_pred, from_logits=True, label_smoothing=0):
        epison = 0.0001

        y_true = tf.cast(
            tf.reshape(
                tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1]),
                (-1, y_pred.shape[-1])
            ),
            y_pred.dtype
        )

        # label smoothing
        if 'label_smooth' in self.compile_params:
            y_true = y_true * (1.0 - self.compile_params['label_smooth']) + (self.compile_params['label_smooth'] / 2.)

        loss = - (y_true * tf.math.log(y_pred + epison) + (1 - y_true) * tf.math.log(1 - y_pred + epison))
        loss = tf.reduce_mean(loss)
        return loss

    def set_callbacks(self):
        """ if using model.fit to train model,
              then we need to set callbacks for the training process """
        # callback for tensorboard
        callback_tf_board = Board(log_dir=self.tb_dir, **self.tb_params)
        callback_tf_board.set_model(self.model)

        # callback for saving model and early stopping
        callback_saver = Saver(self.checkpoint_path, **self.monitor_params)
        callback_saver.set_model(self.model)

        self.callbacks = [callback_tf_board, callback_saver]

    def compile(self):
        loss = self.loss if self.compile_params['customize_loss'] else self.compile_params['loss']
        self.model.compile(optimizer=self.compile_params['optimizer'],
                           loss=loss,
                           metrics=self.compile_params['metrics'])

    @staticmethod
    def __get_best_model_path(model_dir):
        """ get the best model within model_dir """
        file_list = os.listdir(model_dir)
        file_list.sort()
        return os.path.join(model_dir, file_list[-1])

    def load_model(self, model_dir='', x=None, y=None):
        # get best model path
        model_dir = model_dir if model_dir else self.model_dir
        model_path = self.__get_best_model_path(model_dir)

        # empty fit, to prevent error from occurring when loading model
        self.model.fit(x, y, epochs=0) if not isinstance(x, type(None)) else None

        # load model weights
        self.model.load_weights(model_path)
        print(f'Successfully loading weights from {model_path} ')

    def train(self, train_x, train_y, val_x=None, val_y=None):
        # compile model
        self.compile()

        # if we want to load a trained model
        if self.checkpoint_params['load_model']:
            model_dir = path_lib.create_dir_in_root(*(['runtime', 'models'] + self.checkpoint_params['load_model']))
            batch_x = [v[:1] for v in train_x] if isinstance(train_x, tuple) else train_x[:1]
            self.load_model(model_dir, batch_x, train_y[:1])

        if not self.__finish_train:
            # fit model
            self.model.fit(train_x, train_y,
                           epochs=self.train_params['epoch'],
                           batch_size=self.train_params['batch_size'],
                           validation_data=(val_x, val_y) if not isinstance(val_x, type(None)) else None,
                           callbacks=self.callbacks,
                           steps_per_epoch=int(len(train_x) / self.train_params['batch_size']),
                           verbose=2)

            # load the best model so that it could be tested
            self.load_model()

        self.__finish_train = True

    def predict_score(self, x):
        return self.model(x)

    def predict_label(self, x):
        return np.argmax(self.predict_score(x), axis=-1)

    def test(self, x, y):
        """ Evaluate the performance """
        ret = {}

        pred_scores = self.predict_score(x)
        pred_labels = np.argmax(pred_scores, axis=-1)

        for indicator, func in self.evaluate_label_dict.items():
            ret[indicator] = func(y, pred_labels)

        for indicator, func in self.evaluate_score_dict.items():
            ret[indicator] = func(y, pred_scores[:, 1])

        return ret
