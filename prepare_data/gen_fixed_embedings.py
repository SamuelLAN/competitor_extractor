import numpy as np
from scipy.spatial.distance import cdist
from lib import path_lib

# load embeddings
pkl_path = path_lib.get_relative_file_path('runtime', 'input_cache', f'company_embeddings.pkl')
company_embeddings = path_lib.read_cache(pkl_path)
X, names = list(zip(*company_embeddings))
X = np.array(X)
names = np.array(names)

# format embeddings
d_linkedin_name_2_embeddings = {}
for i, embedding in enumerate(X):
    d_linkedin_name_2_embeddings[names[i]] = embedding

# save results
path_lib.cache(path_lib.get_relative_file_path('runtime', 'processed_input', 'd_linkedin_name_2_embeddings.pkl'),
               d_linkedin_name_2_embeddings)

# load the data
json_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v3.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

# calculate the cosine distance
distances = cdist(X, X, 'cosine')

# get the results with the top k minimal cosine distance
for i in range(len(distances)):
    distances[i, i] = 2
similarities = 1 - np.tanh(distances)
top_k_idx = similarities.argsort()[:, -600:]
top_k_idx = top_k_idx[::-1]

# format data
d_linkedin_name_2_similar_names = {}
for i, name in enumerate(names):
    similar_names = list(names[top_k_idx[i]])
    similar_names = list(filter(
        lambda x: f'{min(x, name)}____{max(x, name)}' not in d_min_linkedin_name_max_linkedin_name, similar_names))
    d_linkedin_name_2_similar_names[name] = similar_names

# save results
path_lib.write_json(
    path_lib.get_relative_file_path('runtime', 'processed_input', 'd_linkedin_name_2_similar_names.json'),
    d_linkedin_name_2_similar_names)

print('done')
