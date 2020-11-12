import os
import json
import pandas as pd
from config import path
from lib import format_name

competitor_path = os.path.join(path.DATA_DIR, 'competitor.tsv')

print('\nReading data from competitor csv ... ')
competitor_data = pd.read_csv(competitor_path, delimiter='\t')

# show columns for competitor csv
competitor_columns = list(competitor_data.columns)
for i, v in enumerate(competitor_columns):
    print(f'{i}: {v}: {competitor_data.iloc[0][i]}: {competitor_data.iloc[1][i]}')

# initialize variables
d_gvkey_2_name = {}
d_name_2_gvkey = {}
d_gvkey_pair = {}

print('\nadding gvkey and company name to dictionary ... ')
for gvkey_1, name_1, gvkey_2, name_2 in competitor_data.iloc:
    gvkey_1 = str(gvkey_1)
    gvkey_2 = str(gvkey_2)

    _formatted_name_1 = format_name.company_name(name_1)
    _formatted_name_2 = format_name.company_name(name_2)

    # take the longer name if there are duplicated name
    if gvkey_1 not in d_gvkey_2_name or len(_formatted_name_1.split(' ')[-1]) > 1:
        d_gvkey_2_name[gvkey_1] = _formatted_name_1

    # take the longer name if there are duplicated name
    if gvkey_2 not in d_gvkey_2_name or len(_formatted_name_2.split(' ')[-1]) > 1:
        d_gvkey_2_name[gvkey_2] = _formatted_name_2

    d_name_2_gvkey[_formatted_name_1] = gvkey_1
    d_name_2_gvkey[_formatted_name_2] = gvkey_2

    gvkey_pair = f'{min(gvkey_1, gvkey_2)}_{max(gvkey_1, gvkey_2)}'
    d_gvkey_pair[gvkey_pair] = True

print(f'len of d_gvkey_2_name: {len(d_gvkey_2_name)}')
print(f'len of d_name_2_gvkey: {len(d_name_2_gvkey)}')
print(f'len of d_gvkey_pair: {len(d_gvkey_pair)}')

d_gvkey_2_domain = {}

print(f'\nloading public firms mapped gvkey ...')
gvkey_map_dir = os.path.join(path.DATA_DIR, 'public_companies_firmid_gvkeys_yearwise')
for file_name in os.listdir(gvkey_map_dir):
    file_path = os.path.join(gvkey_map_dir, file_name)
    tmp_data = pd.read_csv(file_path, delimiter='\t')

    for val in tmp_data.iloc:
        domain = format_name.domain(str(val[0]))
        gvkey = str(val[2])
        if not domain or pd.isna(domain):
            continue

        d_gvkey_2_domain[gvkey] = domain

print(f'len of d_gvkey_2_domain: {len(d_gvkey_2_domain)}')

names_domains = []
names_no_domain = []

# check how many companies have domains
for gvkey, name in d_gvkey_2_name.items():
    if gvkey in d_gvkey_2_domain:
        names_domains.append([name, d_gvkey_2_domain[gvkey], gvkey])

    else:
        names_no_domain.append([name, gvkey])

print(f'\ncount of gvkeys that can be mapped in the public firms file: {len(names_domains)}')
print(f'count of gvkeys that cannot be mapped in the public firms file: {len(names_no_domain)}')

linkedin_path = os.path.join(path.DATA_DIR, 'dict_linkedin_url_2_linkedin_v2.json')

print('\nloading linkedin data ...')
with open(linkedin_path, 'rb') as f:
    d_linkedin_url_2_linkedin_val = json.load(f)

d_domain_2_linkedin_val = {}
d_name_2_linkedin_val = {}
d_linkedin_name_2_linkedin_val = {}

print('formatting linkedin data ... ')
for _, linkedin_val in d_linkedin_url_2_linkedin_val.items():
    if 'main' not in linkedin_val or not linkedin_val['main']:
        continue

    main_val = linkedin_val['main']
    if 'website' not in main_val or not main_val['website'] or \
            'name' not in main_val or not main_val['name'] or \
            'description' not in main_val or not main_val['description']:
        continue

    website = format_name.domain(main_val['website'])
    _formatted_name = format_name.company_name(main_val['name'])

    d_domain_2_linkedin_val[website] = linkedin_val
    d_name_2_linkedin_val[_formatted_name] = linkedin_val
    d_linkedin_name_2_linkedin_val[main_val['name']] = linkedin_val

print(f'len of d_domain_2_linkedin_val: {len(d_domain_2_linkedin_val)}')
print(f'len of d_name_2_linkedin_val: {len(d_name_2_linkedin_val)}')

d_gvkey_2_linkedin_val = {}

print('\ncalculating statistic for inter sec between linkedin and competitor data ... ')
count_match_domain = 0
count_match_domain_name = 0
domains_to_be_searched = []

for name, domain, gvkey in names_domains:
    if domain in d_domain_2_linkedin_val:
        count_match_domain += 1
        count_match_domain_name += 1
        d_gvkey_2_linkedin_val[gvkey] = d_domain_2_linkedin_val[domain]

    elif name in d_name_2_linkedin_val:
        count_match_domain_name += 1
        d_gvkey_2_linkedin_val[gvkey] = d_name_2_linkedin_val[name]

    else:
        domains_to_be_searched.append([domain, name])

for name, gvkey in names_no_domain:
    if name not in d_name_2_linkedin_val:
        continue

    count_match_domain_name += 1
    d_gvkey_2_linkedin_val[gvkey] = d_name_2_linkedin_val[name]

print(f'count_match_domain: {count_match_domain}')
print(f'count_match_domain_name: {count_match_domain_name}')

competitors = []
names_to_be_search = []
d_min_gvkey_max_gvkey = {}
d_gvkey = {}

d_min_linkedin_name_max_linkedin_name = {}

print('\nExtracting competitor linkedin data ... ')
for gvkey_1, name_1, gvkey_2, name_2 in competitor_data.iloc:
    gvkey_1 = str(gvkey_1)
    gvkey_2 = str(gvkey_2)

    # if no data, put it to the search queue "names_to_be_search"
    if gvkey_1 not in d_gvkey_2_linkedin_val:
        names_to_be_search.append(name_1)

    if gvkey_2 not in d_gvkey_2_linkedin_val:
        names_to_be_search.append(name_2)

    # if there are linkedin data for the competitor relationship
    if gvkey_1 in d_gvkey_2_linkedin_val and gvkey_2 in d_gvkey_2_linkedin_val:
        # to remove duplicate data
        key = f'{min(gvkey_1, gvkey_2)}_{max(gvkey_1, gvkey_2)}'
        if key in d_min_gvkey_max_gvkey:
            continue
        d_min_gvkey_max_gvkey[key] = True
        d_gvkey[gvkey_1] = True
        d_gvkey[gvkey_2] = True

        competitors.append([
            d_gvkey_2_linkedin_val[gvkey_1],
            d_gvkey_2_linkedin_val[gvkey_2],
        ])

        name_1 = d_gvkey_2_linkedin_val[gvkey_1]['main']['name']
        name_2 = d_gvkey_2_linkedin_val[gvkey_2]['main']['name']

        key = f'{min(name_1, name_2)}____{max(name_1, name_2)}'
        d_min_linkedin_name_max_linkedin_name[key] = True

print(f'\ncount of competitor relationships: {len(competitors)}')
print(f'count of distinct competitor companies: {len(d_gvkey)}')

print('\nsaving data ...')

with open(os.path.join(path.DATA_DIR, 'runtime', 'competitor_linkedin.json'), 'wb') as f:
    f.write(json.dumps(competitors).encode('utf-8'))

with open(os.path.join(path.DATA_DIR, 'runtime', 'competitor_linkedin_dict_format_v3.json'), 'wb') as f:
    f.write(json.dumps({
        'd_linkedin_name_2_linkedin_val': d_linkedin_name_2_linkedin_val,
        'd_min_linkedin_name_max_linkedin_name': d_min_linkedin_name_max_linkedin_name,
    }).encode('utf-8'))

search_texts = list(set(names_to_be_search)) + list(map(lambda x: x[0], domains_to_be_searched))
confirm_names = list(set(names_to_be_search)) + list(map(lambda x: x[1], domains_to_be_searched))

df = pd.DataFrame({'search_text': search_texts, 'company_name': confirm_names})
df.to_csv(os.path.join(path.DATA_DIR, 'runtime', 'to_be_search.csv'), index=False)

print('\ndone')

# output:
# adding gvkey and company name to dictionary ...
# len of d_gvkey_2_name: 3358
# len of d_name_2_gvkey: 5230
#
# loading public firms mapped gvkey ...
# len of d_gvkey_2_domain: 11181
#
# count of gvkeys that can be mapped in the public firms file: 2838
# count of gvkeys that cannot be mapped in the public firms file: 520
#
# loading linkedin data ...
# formatting linkedin data ...
# len of d_domain_2_linkedin_val: 6752
# len of d_name_2_linkedin_val: 6801
#
# calculating statistic for inter sec between linkedin and competitor data ...
# count_match_domain: 1461
# count_match_domain_name: 1799
#
# Extracting competitor linkedin data ...
#
# count of competitor relationships: 9662
# count of distinct competitor companies: 1725
