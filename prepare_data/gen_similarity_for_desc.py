import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from lib import path_lib
from config.path import VERSION

print('\nloading the embeddings ... ')

# load embeddings
pkl_path = path_lib.get_relative_file_path('runtime', 'input_cache', f'company_embeddings_{VERSION}.pkl')
company_embeddings = path_lib.read_cache(pkl_path)
X, names = list(zip(*company_embeddings))
X = np.array(X)
names = np.array(names)

print('\ncalculating the cosine distance ... ')

# calculate the cosine distance
distances = cdist(X, X, 'cosine')

# get the results with the top k minimal cosine distance
similarities = 1 - np.tanh(distances)

print('\nloading the competitor data ... ')

# load linkedin data
json_path = path_lib.get_relative_file_path('runtime', f'competitor_linkedin_dict_format_{VERSION}.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

print('\nsaving the similarities for each company pairs')

d_min_name_max_name_2_sim = {}

length = len(names)
for i, name_1 in enumerate(names):

    if i % 2 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    for j, name_2 in enumerate(names):
        if i == j:
            continue

        min_name = min(name_1, name_2)
        max_name = max(name_1, name_2)

        min_val = d_linkedin_name_2_linkedin_val[min_name]
        max_val = d_linkedin_name_2_linkedin_val[max_name]

        min_website = min_val['main']['website'] if 'main' in min_val and 'website' in min_val['main'] else ''
        max_website = max_val['main']['website'] if 'main' in max_val and 'website' in max_val['main'] else ''

        key = f'{min_name}____{max_name}'
        is_competitor = 'competitor' if key in d_min_linkedin_name_max_linkedin_name else 'non-competitor'

        try:
            d_min_name_max_name_2_sim[key] = [
                min_name, min_website, max_name, max_website, similarities[i][j], is_competitor
            ]
        except:
            pass

data = list(map(list, d_min_name_max_name_2_sim.values()))
df = pd.DataFrame(data, columns=['name_1', 'website_1', 'name_2', 'website_2', 'similarity', 'is_competitor'])
df.to_csv(path_lib.get_relative_file_path('runtime', 'result_csv', f'linkedin_similarity_{VERSION}.csv'), index=False)

# statistic the competitor data
competitor_data = list(filter(lambda x: x[-1] == 'competitor', data))

print(f'\n\ncount of competitor relationship: {len(competitor_data)}')
print(f'mean similarity for competitors: {np.mean(list(map(lambda x: x[-2], competitor_data)))}')
print(f'std similarity for competitors: {np.std(list(map(lambda x: x[-2], competitor_data)))}')
print(f'max similarity for competitors: {np.max(list(map(lambda x: x[-2], competitor_data)))}')
print(f'min similarity for competitors: {np.min(list(map(lambda x: x[-2], competitor_data)))}')

print('\ndone')
