import os
import numpy as np
import random
from lib import path_lib


class Loader:
    """
    Load the dataset
        the dataset consists of "company pairs"
        (the negative samples are choosed from the most similar 500 companies)

    Compare to version 1, the intersection between the train set and the validation and test set are almost empty

    Compare to version 2, the negative_rate is the times of the positive rate
                (in version 2, the negative_rate (will be times 2 automatically) is two times of the positive rate)
    """

    def __init__(self, negative_rate=1, start_ratio=0.0, end_ratio=0.81, use_cache=True):
        self.__competitor_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v3.json')
        self.__embedding_path = path_lib.get_relative_file_path(
            'runtime', 'processed_input', 'd_linkedin_name_2_embeddings.pkl')
        self.__similar_company_path = path_lib.get_relative_file_path(
            'runtime', 'processed_input', 'd_linkedin_name_2_similar_names.json')

        self.__negative_rate = negative_rate
        self.__start_ratio = start_ratio
        self.__end_ratio = end_ratio

        self.__data = self.__load(use_cache)

    def __load(self, use_cache):
        """ Load the data as embeddings """

        cache_path = path_lib.get_relative_file_path(
            'runtime', 'input_cache',
            f'similar_version_neg_rate_{self.__negative_rate}_start_{self.__start_ratio}_end_{self.__end_ratio}.pkl')
        if use_cache and os.path.isfile(cache_path):
            return path_lib.read_cache(cache_path)

        print(f'\nloading data from {self.__competitor_path} ...')
        tmp = path_lib.load_json(self.__competitor_path)
        d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
        d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

        self.__d_linkedin_name_2_embedding = path_lib.load_pkl(self.__embedding_path)
        self.__d_linkedin_name_2_similar_names = path_lib.load_json(self.__similar_company_path)

        print('splitting dataset ...')
        name_pairs = list(d_min_linkedin_name_max_linkedin_name.keys())
        name_pairs.sort()

        total_pairs = len(name_pairs)
        start_index = int(total_pairs * self.__start_ratio)
        end_index = int(total_pairs * self.__end_ratio)
        name_pairs = name_pairs[start_index: end_index]

        print('generating the positive and negative competitor relationships ... ')

        data = []

        for min_name_max_name in name_pairs:
            name_1, name_2 = min_name_max_name.split('____')

            # get features
            feature_1 = self.__choose_features(name_1, d_linkedin_name_2_linkedin_val[name_1])
            feature_2 = self.__choose_features(name_2, d_linkedin_name_2_linkedin_val[name_2])

            # add positive competitor relationship
            data.append([feature_1, feature_2, 1, name_1, name_2])

            # add negative competitor relationship
            for i in range(int(self.__negative_rate * 2)):
                if random.randint(0, 1) == 0:
                    # randomly choose negative competitor relationship
                    name_2_neg = self.__random_choose(name_1)
                    feature_2_neg = self.__choose_features(name_2_neg, d_linkedin_name_2_linkedin_val[name_2_neg])
                    data.append([feature_1, feature_2_neg, 0, name_1, name_2_neg])

                else:
                    # randomly choose negative competitor relationship
                    name_1_neg = self.__random_choose(name_2)
                    feature_1_neg = self.__choose_features(name_1_neg, d_linkedin_name_2_linkedin_val[name_1_neg])
                    data.append([feature_1_neg, feature_2, 0, name_1_neg, name_2])

        print('shuffling the data ...')
        random.shuffle(data)

        print('writing cache ...')
        path_lib.cache(cache_path, data)

        print('finish loading ')
        return data

    def __choose_features(self, name, linkedin_val):
        """ Define which features are used for the prediction """
        return self.__d_linkedin_name_2_embedding[name]

    def __random_choose(self, name_1):
        """ Randomly choose a negative sample """
        similar_names = self.__d_linkedin_name_2_similar_names[name_1]
        return random.sample(similar_names, 1)[0]

    def all(self):
        """ get complete dataset """
        X1, X2, Y, names_1, names_2 = list(zip(*self.__data))
        return np.array(X1), np.array(X2), np.array(Y, dtype=np.int32), names_1, names_2
