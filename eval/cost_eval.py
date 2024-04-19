import os, csv, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
patterns = ['/', '\\', '|', '*', '-', '+', 'x', 'o', '.']

def add_pattern_to_boxes(ax, patterns):
    for patch, pattern in zip(ax.artists, patterns):
        patch.set_hatch(pattern)


with open('data/results/mind2web/test_domain_tokens.json', 'r') as infile:
    data1 = json.load(infile)
with open('data/results/mind2web/test_task_tokens.json', 'r') as infile:
    data2 = json.load(infile)
with open('data/results/mind2web/test_website_tokens.json', 'r') as infile:
    data3 = json.load(infile)
    
# dict_keys(['process_screenshot', 'page_context', 'limit_elements', 'end_state', 'text_field', 'select_option', 'raw_html', 'mind2web_html', 'filtered_html', 'clean_html', 'limited_html', 'element_filtering', 'makes_sense', 'action_element_selection'])

html_data = {'raw_html': [], 'mind2web_html': [], 'filtered_html': [], 'clean_html': [], 'limited_html': []}
html_data['raw_html'].extend(data1['raw_html'])
html_data['raw_html'].extend(data2['raw_html'])
html_data['raw_html'].extend(data3['raw_html'])
html_data['mind2web_html'].extend(data1['mind2web_html'])
html_data['mind2web_html'].extend(data2['mind2web_html'])
html_data['mind2web_html'].extend(data3['mind2web_html'])
html_data['filtered_html'].extend(data1['filtered_html'])
html_data['filtered_html'].extend(data2['filtered_html'])
html_data['filtered_html'].extend(data3['filtered_html'])
html_data['clean_html'].extend(data1['clean_html'])
html_data['clean_html'].extend(data2['clean_html'])
html_data['clean_html'].extend(data3['clean_html'])
html_data['limited_html'].extend(data1['limited_html'])
html_data['limited_html'].extend(data2['limited_html'])
html_data['limited_html'].extend(data3['limited_html'])

df = pd.DataFrame(html_data)
print(df.median())
exit()

new_data = {'Component': [], 'Input (Tokens)': [], 'Output (Tokens)': []}
for key, val in data1.items():
    if key not in ['raw_html', 'mind2web_html', 'filtered_html', 'clean_html', 'limited_html']:
        for i in range(len(val['input'])):
            new_data['Component'].append(key)
        new_data['Input (Tokens)'].extend(val['input'])
        new_data['Output (Tokens)'].extend(val['output'])
for key, val in data2.items():
    if key not in ['raw_html', 'mind2web_html', 'filtered_html', 'clean_html', 'limited_html']:
        for i in range(len(val['input'])):
            new_data['Component'].append(key)
        new_data['Input (Tokens)'].extend(val['input'])
        new_data['Output (Tokens)'].extend(val['output'])
for key, val in data3.items():
    if key not in ['raw_html', 'mind2web_html', 'filtered_html', 'clean_html', 'limited_html']:
        for i in range(len(val['input'])):
            new_data['Component'].append(key)
        new_data['Input (Tokens)'].extend(val['input'])
        new_data['Output (Tokens)'].extend(val['output'])

# -1105, +765
df = pd.DataFrame(new_data)


replace_dict = {'process_screenshot': 'Process Screenshot', 'page_context': 'Page Context', 'filter_elements': 'Filter Elements', 'limit_elements': 'Limiting Strings', 'element_filtering': 'Candidate Proposal', 'makes_sense': 'Double Checking', 'action_element_selection': 'Element Action Selection', 'end_state': 'End State', 'text_field': 'Text Field', 'select_option': 'Select Option'}

df = df[df['Component'] != 'select_option']

cost_info = {
    "GPT-3.5-Turbo": (0.5, 1.5),
    "GPT-3.5-Long": (0.5, 1.5),
    "GPT-4-Turbo": (10, 30),
    "GPT-4-Vision": (10, 30),
}

component_map = {
    'Candidate Proposal': 'GPT-3.5-Turbo',
    'Element Action Selection': 'GPT-4-Turbo',
    'Double Checking': 'GPT-4-Turbo',
    'Secondary Parameter': 'GPT-4-Turbo',
    'High Level': 'GPT-3.5-Turbo',
    'End State': 'GPT-4-Turbo',
    'Page Context': 'GPT-3.5-Turbo',
    'Select Option': 'GPT-4-Turbo',
    'Process Screenshot': 'GPT-4-Vision',
    'Limiting Strings': 'GPT-3.5-Turbo',
    'Text Field': 'GPT-4-Turbo'
}

def calculate_cost(row):
    component = row['Component']
    input_tokens = row['Input (Tokens)']
    output_tokens = row['Output (Tokens)']
    
    # Lookup the model from the component map
    model = component_map.get(component)
    
    if model:
        # Lookup the cost info based on the model
        cost_types = cost_info.get(model)
        if cost_types:
            # Calculate cost based on the cost info and tokens
            if component == 'Process Screenshot':
                input_tokens = input_tokens - 1105 + 765
            cost = (cost_types[0] * input_tokens / 1000000) + (cost_types[1] * output_tokens / 1000000)
            return cost
    
    return None

df['Component'] = df['Component'].replace(replace_dict)
df['Cost ($)'] = df.apply(calculate_cost, axis=1)

for component in df['Component'].unique():
    print(component)
    print(df['Cost ($)'].where(df['Component'] == component).median())

plt.figure()
sns.set_theme(style='whitegrid', font_scale=1.2)
# plt.figure(figsize=(12, 6))
g = sns.boxplot(x='Component', y="Cost ($)", data=df, showfliers=False, order=['Process Screenshot', 'Page Context', 'Limiting Strings', 'Candidate Proposal', 'Double Checking', 'Element Action Selection', 'Text Field', 'End State'])
plt.xlabel('Component')
plt.ylabel('Cost ($)')
# plt.yscale('log')
# plt.ylabel("Tokens")
plt.title('Component Costs')
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig('data/plots/component_costs.pdf')
exit()

