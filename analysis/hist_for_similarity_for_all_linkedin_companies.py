import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib import path_lib

# load the similarity data
similarity_csv_path = path_lib.get_relative_file_path('runtime', 'result_csv', 'linkedin_similarity.csv')
similarity_csv = pd.read_csv(similarity_csv_path)

similarities = list(similarity_csv['similarity'])
competitor_similarities = list(similarity_csv[similarity_csv['is_competitor'] == 'competitor']['similarity'])

print(f'\n\ncount of all company pairs: {len(similarities)}')
print(f'mean similarity for all company pairs: {np.mean(similarities)}')
print(f'std similarity for all company pairs: {np.std(similarities)}')
print(f'max similarity for all company pairs: {np.max(similarities)}')
print(f'min similarity for all company pairs: {np.min(similarities)}')

print(f'\ncount of competitor relationship: {len(competitor_similarities)}')
print(f'mean similarity for competitors: {np.mean(competitor_similarities)}')
print(f'std similarity for competitors: {np.std(competitor_similarities)}')
print(f'max similarity for competitors: {np.max(competitor_similarities)}')
print(f'min similarity for competitors: {np.min(competitor_similarities)}')

plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.hist(similarities, bins=30, edgecolor='white')
plt.title('histogram for similarity of all company pairs', fontsize=22)
plt.xlabel('similarity', fontsize=16)
plt.ylabel('count of company pairs', fontsize=16)
plt.xticks(list(np.linspace(0.1, 1., 10)), fontsize=14)
plt.xlim(0.1, 1.)
# plt.savefig(
#     path_lib.get_relative_file_path('runtime', 'analysis', 'figures', 'hist_for_similarity_for_all_company_pairs.png'),
#     dpi=300)
# plt.show()
# plt.close()
#
# plt.figure(figsize=(14, 8))
plt.subplot(212)
plt.hist(competitor_similarities, bins=30, edgecolor='white')
plt.title('histogram for similarity of the competitors', fontsize=22)
plt.xlabel('similarity', fontsize=16)
plt.ylabel('count of competitor pairs', fontsize=16)
plt.xticks(list(np.linspace(0.1, 1., 10)), fontsize=14)
plt.xlim(0.1, 1.)
plt.subplots_adjust(hspace=0.45)
plt.savefig(
    # path_lib.get_relative_file_path('runtime', 'analysis', 'figures', 'hist_for_similarity_for_competitors.png'),
    path_lib.get_relative_file_path('runtime', 'analysis', 'figures', 'hist_for_similarity.png'),
    dpi=300)
plt.show()
plt.close()

print('\ndone')

"""
count of all company pairs: 45167760
mean similarity for all company pairs: 0.4694795092543445
std similarity for all company pairs: 0.1169462500073105
max similarity for all company pairs: 1.0
min similarity for all company pairs: 0.15235062352614026

count of competitor relationship: 20326
mean similarity for competitors: 0.5759494816108272
std similarity for competitors: 0.12464365426374166
max similarity for competitors: 0.9044875194205836
min similarity for competitors: 0.20937702813234335
"""
