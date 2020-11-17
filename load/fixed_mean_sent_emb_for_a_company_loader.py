import os
import json
import numpy as np
from config.path import VERSION
from lib import path_lib
from nltk.tokenize import sent_tokenize


class Loader:
    """
    Load the dataset
        the dataset consists of company embeddings instead of company pairs
    """

    def __init__(self, use_cache=True):
        self.__competitor_path = path_lib.get_relative_file_path('runtime', f'competitor_linkedin_dict_format_{VERSION}.json')
        self.__data = self.__load(use_cache)

    def __load(self, use_cache):
        """ Load the data as embeddings """

        cache_path = path_lib.get_relative_file_path('runtime', 'input_cache', f'company_embeddings_{VERSION}.pkl')
        if use_cache and os.path.isfile(cache_path):
            return path_lib.read_cache(cache_path)

        print(f'\nloading data from {self.__competitor_path} ...')
        with open(self.__competitor_path, 'rb') as f:
            tmp = json.load(f)
        d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']

        data = []

        print('loading sentence bert to generate embeddings ...')
        from sentence_transformers import SentenceTransformer
        self.__sentence_bert = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

        # converting the raw data to features that we need
        for linkedin_name, linkedin_val in d_linkedin_name_2_linkedin_val.items():
            # get features
            feature = self.__choose_features(linkedin_val)
            data.append([feature, linkedin_name])

        print('writing cache ...')
        path_lib.cache(cache_path, data)

        print('finish loading ')
        return data

    def __choose_features(self, linkedin_val):
        """ Define which features are used for the prediction """
        description = linkedin_val['main']['description']
        sentences = sent_tokenize(description)
        return np.mean(self.__sentence_bert.encode(sentences), axis=0)

    def all(self):
        """ get complete dataset """
        X, names = list(zip(*self.__data))
        return np.array(X), np.array(names)


# o_loader = Loader(use_cache=False)
# X, names = o_loader.all()
# print(f'X.shape: {X.shape}')
