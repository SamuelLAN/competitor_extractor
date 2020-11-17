from lib import path_lib

# load the json data
json_path = path_lib.get_relative_file_path('runtime', 'competitor_linkedin_dict_format_v4.json')
tmp = path_lib.load_json(json_path)
d_linkedin_name_2_linkedin_val = tmp['d_linkedin_name_2_linkedin_val']
d_min_linkedin_name_max_linkedin_name = tmp['d_min_linkedin_name_max_linkedin_name']

# for remove duplicated statistics
d_name_2_has_statistic = {}
pub_num = 0

# traverse all the companies to statistic the public companies
name_pairs = list(d_min_linkedin_name_max_linkedin_name.keys())
for min_name_max_name in name_pairs:
    name_1, name_2 = min_name_max_name.split('____')

    if name_1 not in d_name_2_has_statistic:
        d_name_2_has_statistic[name_1] = True

        linkedin_val = d_linkedin_name_2_linkedin_val[name_1]
        if 'main' in linkedin_val and 'company_type' in linkedin_val['main'] and \
                linkedin_val['main']['company_type'].lower() == 'public company':
            pub_num += 1

    if name_2 not in d_name_2_has_statistic:
        d_name_2_has_statistic[name_2] = True

        linkedin_val = d_linkedin_name_2_linkedin_val[name_2]
        if 'main' in linkedin_val and 'company_type' in linkedin_val['main'] and \
                linkedin_val['main']['company_type'].lower() == 'public company':
            pub_num += 1

print(f'count of public companies: {pub_num}')

# count of public companies: 2142
