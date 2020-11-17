import os
import json
import numpy as np
import random
from lib import path_lib
from config.path import VERSION
from nltk.tokenize import sent_tokenize


class Loader:
    """
    Load the dataset
        the dataset consists of "company pairs"
    """

    def __init__(self, negative_rate=1, use_cache=True):
        self.__competitor_path = path_lib.get_relative_file_path('runtime', f'competitor_linkedin_dict_format_{VERSION}.json')
        self.__negative_rate = negative_rate

        self.__data = self.__load(use_cache)

    def __load(self, use_cache):
        """ Load the data as embeddings """

        cache_path = path_lib.get_relative_file_path('runtime', 'input_cache',
                                                     f'neg_rate_{self.__negative_rate}_v2.pkl')
        if use_cache and os.path.isfile(cache_path):
            return path_lib.read_cache(cache_path)

        print(f'\nloading data from {self.__competitor_path} ...')
        tmp = path_lib.load_json(self.__competitor_path)
        d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
        d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

        names = list(d_linkedin_name_2_linkedin_val.keys())

        print('generating the positive and negative competitor relationships ... ')

        data = []

        print('loading sentence bert to generate embeddings ...')
        from sentence_transformers import SentenceTransformer
        self.__sentence_bert = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

        for min_name_max_name in d_min_linkedin_name_max_linkedin_name.keys():
            name_1, name_2 = min_name_max_name.split('____')

            # get features
            feature_1 = self.__choose_features(d_linkedin_name_2_linkedin_val[name_1])
            feature_2 = self.__choose_features(d_linkedin_name_2_linkedin_val[name_2])

            # add positive competitor relationship
            data.append([feature_1, feature_2, 1, name_1, name_2])

            # add negative competitor relationship
            for i in range(self.__negative_rate):
                # randomly choose negative competitor relationship
                name_2_neg = self.__random_choose(names, name_1, d_min_linkedin_name_max_linkedin_name)
                feature_2_neg = self.__choose_features(d_linkedin_name_2_linkedin_val[name_2_neg])

                # randomly choose negative competitor relationship
                name_1_neg = self.__random_choose(names, name_2, d_min_linkedin_name_max_linkedin_name)
                feature_1_neg = self.__choose_features(d_linkedin_name_2_linkedin_val[name_1_neg])

                data.append([feature_1, feature_2_neg, 0, name_1, name_2_neg])
                data.append([feature_1_neg, feature_2, 0, name_1_neg, name_2])

        print('shuffling the data ...')
        random.shuffle(data)

        print('writing cache ...')
        path_lib.cache(cache_path, data)

        print('finish loading ')
        return data

    def __choose_features(self, linkedin_val):
        """ Define which features are used for the prediction """
        description = linkedin_val['main']['description']
        sentences = sent_tokenize(description)
        return np.mean(self.__sentence_bert.encode(sentences), axis=0)

    @staticmethod
    def __random_choose(names, name_1, d_names):
        """ Randomly choose a negative sample """
        name_2 = random.sample(names, 1)[0]
        while name_1 == name_2 or f'{min(name_1, name_2)}____{max(name_1, name_2)}' in d_names:
            name_2 = random.sample(names, 1)[0]
        return name_2

    def all(self):
        """ get complete dataset """
        X1, X2, Y, names_1, names_2 = list(zip(*self.__data))
        return np.array(X1), np.array(X2), np.array(Y, dtype=np.int32), names_1, names_2

    def train_val_test(self, train_ratio, val_ratio):
        """ Get training, validation, and test set """

        # calculate the size of different dataset
        total_len = len(self.__data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)

        # calculate the boundary of different dataset
        train_bound = int(train_len / 64) * 64
        val_bound = train_bound + int(val_len / 64) * 64
        test_bound = val_bound + int((total_len - val_bound) / 64) * 64

        # split data
        ret = self.all()
        return [np.array(v[:train_bound]) for v in ret], \
               [np.array(v[train_bound: val_bound]) for v in ret], \
               [np.array(v[val_bound:test_bound]) for v in ret]

# o_loader = Loader(use_cache=True)
# X1, X2, Y, names_1, names_2 = o_loader.all()
#
# print('\n--------------------------------------')
# print(X1.shape)
# print(X2.shape)
# print(Y.shape)
