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

logs.MODEL = 'statistics'
logs.VARIANT = 'shared_competitor_among_competitors'


def statistic():
    print(f'\nstatistic the shared competitors for competitors ... ')

    # record statistics
    shared_competitor_counts = []

    # to remove duplicate statistic
    d_min_name_max_name_2_has_statistic = {}

    length = len(d_name_2_competitors)
    _i = 0
    for _name_1, competitors in d_name_2_competitors.items():
        if _i % 2 == 0:
            progress = float(_i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')
        _i += 1

        for _j, _name_2 in enumerate(list(competitors)):
            # remove duplicate statistic
            key = f'{min(_name_1, _name_2)}____{max(_name_1, _name_2)}'
            if key in d_min_name_max_name_2_has_statistic:
                continue
            d_min_name_max_name_2_has_statistic[key] = True

            if _name_2 not in d_name_2_competitors:
                shared_competitor_counts.append(0)
                continue

            shared_num = len(competitors.intersection(d_name_2_competitors[_name_2]))
            shared_competitor_counts.append(shared_num)

    logs.new_line()
    logs.add('statistics', 'total count of competitors companies', f'{len(d_name_2_competitors)}', output=True)
    logs.add('statistics', 'mean of shared competitors', f'among competitors: {np.mean(shared_competitor_counts)}',
             output=True)
    logs.add('statistics', 'std of shared competitors', f'among competitors: {np.std(shared_competitor_counts)}',
             output=True)
    logs.add('statistics', 'max of shared competitors', f'among competitors: {np.max(shared_competitor_counts)}',
             output=True)
    logs.add('statistics', 'min of shared competitors', f'among competitors: {np.min(shared_competitor_counts)}',
             output=True)

    bins = list(range(0, 53, 1))
    plt.figure(figsize=(18, 8))
    plt.hist(shared_competitor_counts, bins=bins, edgecolor='white')
    plt.title(
        f'histogram for count of shared competitors among competitors',
        fontsize=22)
    plt.xlabel('count of shared competitors for each similar company pair', fontsize=16)
    plt.ylabel('count of company pairs', fontsize=16)
    plt.xticks(bins)
    plt.savefig(path_lib.get_relative_file_path(
        'runtime', 'analysis', 'figures', f'hist_for_shared_competitor_among_competitors.png'),
        dpi=300)
    plt.show()
    plt.close()


statistic()

print('\ndone')
