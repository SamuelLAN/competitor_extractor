import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib import path_lib
from lib import format_name

# load the json data
json_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v4.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

# get all the competitor names
names = list(map(lambda x: x.split('____'), d_min_linkedin_name_max_linkedin_name.keys()))
names_1, names_2 = list(zip(*names))
names = list(set(names_1 + names_2))
path_lib.write_json(path_lib.get_relative_file_path('runtime', 'competitor_names.json'), names)

public_csv = pd.read_csv(path_lib.get_relative_file_path('QualityControl_2016_result.csv'))
public_data = np.array(public_csv)
columns = list(public_csv.columns)

print('\nexamples of the QualityControl_2016_result')
for i, v in enumerate(columns):
    print(f'{i} : {v} : {public_data[0][i]} : {public_data[1][i]}')

d_format_name_2_sector = {}
d_format_website_2_sector = {}

for i, v in enumerate(public_data):
    name = v[1]
    website = v[10]
    sector = v[9]

    if not sector or pd.isna(sector):
        continue

    if name and pd.notna(name):
        d_format_name_2_sector[format_name.company_name(name)] = sector
    if website and pd.notna(website):
        d_format_name_2_sector[format_name.domain(website)] = sector

print(f'\ncount of competitor companies: {len(names)}')

linkedin_vals = list(map(lambda x: d_linkedin_name_2_linkedin_val[x]['main'], names))

d_industry_2_names = {}
d_sector_2_names = {}

for i, name in enumerate(names):
    linkedin_val = linkedin_vals[i]

    _format_name = format_name.company_name(name)
    _format_website = format_name.domain(linkedin_val['website']) if 'website' in linkedin_val else ''

    if _format_website and _format_website in d_format_website_2_sector:
        sector = d_format_website_2_sector[_format_website]
    elif _format_name in d_format_name_2_sector:
        sector = d_format_name_2_sector[_format_name]
    else:
        sector = ''

    if sector:
        if sector not in d_sector_2_names:
            d_sector_2_names[sector] = set()
        d_sector_2_names[sector].add(name)

    if not linkedin_val or 'industry' not in linkedin_val:
        continue

    for industry in linkedin_val['industry']:
        if industry not in d_industry_2_names:
            d_industry_2_names[industry] = set()
        d_industry_2_names[industry].add(name)

# load linkedin json
d_linkedin_url_2_linkedin_val = path_lib.load_json(
    path_lib.get_relative_file_path('dict_linkedin_url_2_linkedin_v2.json'))
d_industry_2_names_for_all_linkedin = {}

for _, linkedin_val in d_linkedin_url_2_linkedin_val.items():
    if 'main' not in linkedin_val or not linkedin_val['main'] or 'industry' not in linkedin_val['main']:
        continue
    linkedin_val = linkedin_val['main']

    for industry in linkedin_val['industry']:
        if industry not in d_industry_2_names_for_all_linkedin:
            d_industry_2_names_for_all_linkedin[industry] = []
        d_industry_2_names_for_all_linkedin[industry].append(linkedin_val['name'])


def show_statistics(d_key_2_names, title_prefix, show_first_n=39, label_distance=1.1):
    _statistics = list(map(lambda x: [len(x[1]), x[0]], d_key_2_names.items()))
    _statistics.sort(reverse=True)
    _statistics = _statistics[:39] + [[sum(list(map(lambda x: x[0], _statistics[show_first_n:]))), 'other']]

    # print information to the console
    print('\n----------------------------------------------------------')
    print(f'Statistics for {title_prefix}:')
    for count, name in _statistics[:show_first_n]:
        print(f'{name}: {count}')

    # show figures as the pie
    plt.figure(figsize=(16, 8))
    plt.pie(list(map(lambda x: x[0], _statistics)),
            explode=[0] * len(_statistics),
            labels=list(map(lambda x: x[1], _statistics)),
            labeldistance=label_distance,
            autopct='%3.2f%%',
            shadow=False,
            startangle=90,
            pctdistance=0.6
            )
    plt.axis('equal')
    plt.title(f'Pie for the {title_prefix}', fontsize=22)
    plt.legend()
    # plt.savefig(path_lib.get_relative_file_path('runtime', 'analysis', 'figures',
    #                                             f'pie_for_{title_prefix.replace(" ", "_")}.png'), dpi=300)
    plt.show()
    plt.close()


show_statistics(d_industry_2_names, 'industry distribution of the competitors', 39)
show_statistics(d_sector_2_names, 'sector distribution of the competitors', 19)
show_statistics(d_industry_2_names_for_all_linkedin, 'industry distribution of all linkedin companies', 39)

# count of competitor companies: 2604
#
# ----------------------------------------------------------
# Statistics for industry distribution of the competitors:
# Financial Services: 139
# Oil & Energy: 133
# Information Technology and Services: 127
# Biotechnology: 121
# Computer Software: 102
# Medical Devices: 98
# Banking: 93
# Retail: 81
# Pharmaceuticals: 76
# Real Estate: 71
# Internet: 71
# Electrical/Electronic Manufacturing: 71
# Telecommunications: 62
# Semiconductors: 59
# Mining & Metals: 55
# Consumer Goods: 52
# Automotive: 50
# Utilities: 49
# Insurance: 49
# Hospital & Health Care: 45
# Mechanical or Industrial Engineering: 39
# Chemicals: 38
# Construction: 37
# Machinery: 34
# Renewables & Environment: 32
# Food & Beverages: 32
# Building Materials: 30
# Marketing and Advertising: 28
# Health, Wellness and Fitness: 26
# Apparel & Fashion: 25
# Transportation/Trucking/Railroad: 24
# Restaurants: 24
# Investment Management: 23
# Hospitality: 23
# Consumer Electronics: 21
# Logistics and Supply Chain: 20
# Entertainment: 18
# Aviation & Aerospace: 18
# Food Production: 15
#
# ----------------------------------------------------------
# Statistics for sector distribution of the competitors:
# BusSv: 264
# Drugs: 161
# Banks: 138
# Fin: 131
# Chips: 130
# Oil: 84
# Rtail: 79
# MedEq: 79
# Whlsl: 71
# Mach: 71
# Trans: 68
# Insur: 61
# Comps: 60
# Util: 59
# Telcm: 47
# LabEq: 44
# Chems: 43
# Autos: 38
# BldMt: 32
#
# ----------------------------------------------------------
# Statistics for industry distribution of all linkedin companies:
# Machinery: 869
# Mechanical or Industrial Engineering: 525
# Oil & Energy: 449
# Information Technology and Services: 435
# Biotechnology: 418
# Mining & Metals: 412
# Plastics: 385
# Automotive: 371
# Financial Services: 362
# Electrical/Electronic Manufacturing: 353
# Medical Devices: 344
# Computer Software: 294
# Aviation & Aerospace: 285
# Retail: 278
# Pharmaceuticals: 251
# Consumer Goods: 245
# Real Estate: 235
# Construction: 229
# Banking: 228
# Internet: 225
# Marketing and Advertising: 191
# Telecommunications: 186
# Semiconductors: 142
# Hospital & Health Care: 137
# Building Materials: 137
# Defense & Space: 130
# Food & Beverages: 127
# Industrial Automation: 121
# Insurance: 118
# Chemicals: 117
# Restaurants: 109
# Utilities: 108
# Packaging and Containers: 101
# Health, Wellness and Fitness: 91
# Transportation/Trucking/Railroad: 90
# Wholesale: 77
# Apparel & Fashion: 77
# Hospitality: 75
# Food Production: 73
