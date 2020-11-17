import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib import path_lib

# load the json data
json_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v4.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

d_name = {}
for min_name_max_name in d_min_linkedin_name_max_linkedin_name.keys():
    name_1, name_2 = min_name_max_name.split('____')
    d_name[name_1] = True
    d_name[name_2] = True

# load the similarity data
similarity_csv_path = path_lib.get_relative_file_path('runtime', 'result_csv', 'linkedin_similarity.csv')
similarity_csv = pd.read_csv(similarity_csv_path)
similarity_data = np.array(similarity_csv)

similarity_data = list(filter(lambda x: x[0] in d_name or x[2] in d_name, similarity_data))
similarities = list(map(lambda x: x[-2], similarity_data))

competitor_similarities = list(similarity_csv[similarity_csv['is_competitor'] == 'competitor']['similarity'])

print(f'\n\ncount of company pairs that has competitors: {len(similarities)}')
print(f'mean similarity for company pairs that has competitors: {np.mean(similarities)}')
print(f'std similarity for company pairs that has competitors: {np.std(similarities)}')
print(f'max similarity for company pairs that has competitors: {np.max(similarities)}')
print(f'min similarity for company pairs that has competitors: {np.min(similarities)}')

print(f'\ncount of competitor relationship: {len(competitor_similarities)}')
print(f'mean similarity for competitors: {np.mean(competitor_similarities)}')
print(f'std similarity for competitors: {np.std(competitor_similarities)}')
print(f'max similarity for competitors: {np.max(competitor_similarities)}')
print(f'min similarity for competitors: {np.min(competitor_similarities)}')

plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.hist(similarities, bins=30, edgecolor='white')
plt.title('histogram for similarity of company pairs that contain competitor', fontsize=22)
plt.xlabel('similarity', fontsize=16)
plt.ylabel('count of company pairs', fontsize=16)
plt.xticks(list(np.linspace(0.1, 1., 10)), fontsize=14)
plt.xlim(0.1, 1.)
# plt.savefig(
#     path_lib.get_relative_file_path('runtime', 'analysis', 'figures',
#                                     'hist_for_similarity_for_company_pairs_that_has_competitors.png'),
#     dpi=300)
# plt.show()
# plt.close()
plt.subplot(212)
plt.hist(competitor_similarities, bins=30, edgecolor='white')
plt.title('histogram for similarity of the competitors', fontsize=22)
plt.xlabel('similarity', fontsize=16)
plt.ylabel('count of competitor pairs', fontsize=16)
plt.xticks(list(np.linspace(0.1, 1., 10)), fontsize=14)
plt.xlim(0.1, 1.)
plt.subplots_adjust(hspace=0.45)
plt.savefig(
    path_lib.get_relative_file_path('runtime', 'analysis', 'figures', 'hist_for_similarity_for_all_public_companies_and_competitors.png'),
    dpi=300)
plt.show()
plt.close()

print('\ndone')

# count of company pairs that has competitors: 21359310
# mean similarity for company pairs that has competitors: 0.4700309358624736
# std similarity for company pairs that has competitors: 0.11103960043132305
# max similarity for company pairs that has competitors: 1.0
# min similarity for company pairs that has competitors: 0.15235062352614026
#
# count of competitor relationship: 20326
# mean similarity for competitors: 0.5759494816108272
# std similarity for competitors: 0.12464365426374166
# max similarity for competitors: 0.9044875194205836
# min similarity for competitors: 0.20937702813234335
