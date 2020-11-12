import os
import json
import numpy as np
import random
from lib import path_lib
from nltk.tokenize import sent_tokenize


class Loader:
    """
    Load the dataset
        the dataset consists of "company pairs"

    Compare to version 1, the intersection between the train set and the validation and test set are almost empty

    Compare to version 2, the negative_rate is the times of the positive rate
                (in version 2, the negative_rate (will be times 2 automatically) is two times of the positive rate)
    """

    def __init__(self, negative_rate=1, start_ratio=0.0, end_ratio=0.81, use_cache=True):
        self.__competitor_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v3.json')
        self.__negative_rate = negative_rate
        self.__start_ratio = start_ratio
        self.__end_ratio = end_ratio

        self.__data = self.__load(use_cache)

    def __load(self, use_cache):
        """ Load the data as embeddings """

        cache_path = path_lib.get_relative_file_path(
            'runtime', 'input_cache',
            f'neg_rate_{self.__negative_rate}_start_{self.__start_ratio}_end_{self.__end_ratio}.pkl')
        if use_cache and os.path.isfile(cache_path):
            return path_lib.read_cache(cache_path)

        print(f'\nloading data from {self.__competitor_path} ...')
        with open(self.__competitor_path, 'rb') as f:
            tmp = json.load(f)

        d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
        d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

        print('splitting dataset ...')
        name_pairs = list(d_min_linkedin_name_max_linkedin_name.keys())
        name_pairs.sort()

        total_pairs = len(name_pairs)
        start_index = int(total_pairs * self.__start_ratio)
        end_index = int(total_pairs * self.__end_ratio)
        name_pairs = name_pairs[start_index: end_index]

        names = list(d_linkedin_name_2_linkedin_val.keys())

        print('generating the positive and negative competitor relationships ... ')

        data = []

        print('loading sentence bert to generate embeddings ...')
        from sentence_transformers import SentenceTransformer
        self.__sentence_bert = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

        for min_name_max_name in name_pairs:
            name_1, name_2 = min_name_max_name.split('____')

            # get features
            feature_1 = self.__choose_features(d_linkedin_name_2_linkedin_val[name_1])
            feature_2 = self.__choose_features(d_linkedin_name_2_linkedin_val[name_2])

            # add positive competitor relationship
            data.append([feature_1, feature_2, 1, name_1, name_2])

            # add negative competitor relationship
            for i in range(int(self.__negative_rate * 2)):
                if random.randint(0, 1) == 0:
                    # randomly choose negative competitor relationship
                    name_2_neg = self.__random_choose(names, name_1, d_min_linkedin_name_max_linkedin_name)
                    feature_2_neg = self.__choose_features(d_linkedin_name_2_linkedin_val[name_2_neg])
                    data.append([feature_1, feature_2_neg, 0, name_1, name_2_neg])

                else:
                    # randomly choose negative competitor relationship
                    name_1_neg = self.__random_choose(names, name_2, d_min_linkedin_name_max_linkedin_name)
                    feature_1_neg = self.__choose_features(d_linkedin_name_2_linkedin_val[name_1_neg])
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
