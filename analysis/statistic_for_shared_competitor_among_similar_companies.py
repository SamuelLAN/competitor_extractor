import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from lib import path_lib
from lib import logs

print('\nloading the competitor data ... ')

# load the json data
json_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v4.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

print('\nformatting the competitor data structure ... ')

d_name_2_competitors = {}

name_pairs = list(d_min_linkedin_name_max_linkedin_name.keys())
for min_name_max_name in name_pairs:
    name_1, name_2 = min_name_max_name.split('____')

    if name_1 not in d_name_2_competitors:
        d_name_2_competitors[name_1] = set()
    d_name_2_competitors[name_1].add(name_2)

    if name_2 not in d_name_2_competitors:
        d_name_2_competitors[name_2] = set()
    d_name_2_competitors[name_2].add(name_1)

print('\nloading the embeddings ... ')

# load embeddings
pkl_path = path_lib.get_relative_file_path('runtime', 'input_cache', f'company_embeddings.pkl')
company_embeddings = path_lib.read_cache(pkl_path)
X, names = list(zip(*company_embeddings))
X = np.array(X)
names = np.array(names)

print('\ncalculating the cosine distance ... ')

# calculate the cosine distance
distances = cdist(X, X, 'cosine')
for i in range(len(distances)):
    distances[i, i] = 2

# get the results with the top k minimal cosine distance
similarities = 1 - np.tanh(distances)
top_k_idx = similarities.argsort()

logs.MODEL = 'statistics'
logs.VARIANT = 'shared_competitor_among_similar_companies'


def statistic(_top_k_similar):
    _top_k_idx = top_k_idx[:, -_top_k_similar:]
    _top_k_idx = _top_k_idx[::-1]

    print(f'\nstatistic the shared competitors for top {_top_k_similar} similar companies of all Linkedin companies ... ')

    # record statistics
    shared_competitor_counts = []

    # to remove duplicate statistic
    d_min_name_max_name_2_has_statistic = {}

    length = len(names)
    for _i, _name_1 in enumerate(names):

        if _i % 2 == 0:
            progress = float(_i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        similar_names = names[_top_k_idx[_i]]

        for _j, _name_2 in enumerate(similar_names):

            # remove duplicate statistic
            key = f'{min(_name_1, _name_2)}____{max(_name_1, _name_2)}'
            if key in d_min_name_max_name_2_has_statistic:
                continue
            d_min_name_max_name_2_has_statistic[key] = True

            if _name_1 not in d_name_2_competitors or _name_2 not in d_name_2_competitors:
                shared_competitor_counts.append(0)
                continue

            competitor_set_1 = d_name_2_competitors[_name_1]
            competitor_set_2 = d_name_2_competitors[_name_2]

            shared_num = len(competitor_set_1.intersection(competitor_set_2))
            shared_competitor_counts.append(shared_num)

    logs.new_line()
    logs.add('statistics', 'total count of companies', f'{len(names)}', output=True)
    logs.add('statistics', 'mean of shared competitors', f'among top {_top_k_similar} similar companies: {np.mean(shared_competitor_counts)}', output=True)
    logs.add('statistics', 'std of shared competitors', f'among top {_top_k_similar} similar companies: {np.std(shared_competitor_counts)}', output=True)
    logs.add('statistics', 'max of shared competitors', f'among top {_top_k_similar} similar companies: {np.max(shared_competitor_counts)}', output=True)
    logs.add('statistics', 'min of shared competitors', f'among top {_top_k_similar} similar companies: {np.min(shared_competitor_counts)}', output=True)

    num_0 = len(list(filter(lambda x: x == 0, shared_competitor_counts)))
    shared_competitor_counts = list(filter(lambda x: x > 0, shared_competitor_counts))

    plt.figure(figsize=(14, 8))
    plt.hist(shared_competitor_counts, bins=[0.1, 1, 2, 3, 4, 5, 10, 20, 40], edgecolor='white')
    plt.title(
        f'histogram for count of shared competitors among top {_top_k_similar} similar companies of all Linkedin companies\n(spike for ({num_0} zero shared competitors) is removed)',
        fontsize=22)
    plt.xlabel('count of shared competitors for each similar company pair', fontsize=16)
    plt.ylabel('count of company pairs', fontsize=16)
    plt.xticks([0, 1, 2, 3, 4, 5, 10, 20, 40])
    plt.savefig(
        path_lib.get_relative_file_path('runtime', 'analysis', 'figures',
                                        f'hist_for_shared_competitor_among_top_{_top_k_similar}_similar_companies.png'),
        dpi=300)
    plt.show()
    plt.close()


# for top_k_similar in [20, 30, 50, 100, 200, 300, 400, 500]:
for top_k_similar in [20]:
    statistic(top_k_similar)

print('\ndone')
