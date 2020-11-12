import os
import json
from lib import path_lib
import pandas as pd


def parse_log(lines):
    _top_k = []
    _acc = []
    _precision = []
    _recall = []
    _f1 = []

    start = False

    for line in lines:

        if 'data_params : ' in line:
            if '"neg_rate": 1, "neg_rate_train": 1, "neg_rate_val": 1, "neg_rate_test": 3' in line:
                start = True
            else:
                start = False

        if not start:
            continue

        if 'model_params : ' in line:
            tmp_data = line[line.index('{'): line.index('}') + 1]
            tmp_data = json.loads(tmp_data)
            _top_k.append(tmp_data['top_k'])

        find_string = 'test_evaluation : acc: '
        if find_string in line:
            _acc.append(line[line.index(find_string) + len(find_string):])

        find_string = 'test_evaluation : precision: '
        if find_string in line:
            _precision.append(line[line.index(find_string) + len(find_string):])

        find_string = 'test_evaluation : recall: '
        if find_string in line:
            _recall.append(line[line.index(find_string) + len(find_string):])

        find_string = 'test_evaluation : f1: '
        if find_string in line:
            _f1.append(line[line.index(find_string) + len(find_string):])

    return _top_k, _acc, _precision, _recall, _f1


data = []
log_dir = path_lib.create_dir_in_root('log', 'fixed_mean_sent_emb_similarity')

for file_name in os.listdir(log_dir):
    file_path = os.path.join(log_dir, file_name)

    with open(file_path, 'rb') as f:
        content = f.readlines()

    content = list(map(lambda x: x.decode('utf-8').strip(), content))
    data += list(zip(*parse_log(content)))

data = list(map(list, data))
data.sort()

df = pd.DataFrame(data, columns=['top_k', 'acc', 'precision', 'recall', 'f1'])
df.to_csv(path_lib.get_relative_file_path('runtime', 'result_csv', 'test_similarity.csv'), index=False)

print('\ndone')
