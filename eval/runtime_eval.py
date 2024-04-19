import os, csv, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
patterns = ['//', '\\\\', '||', '**', '--', '++', 'xx', 'OO', '..']

def add_pattern_to_boxes(ax, patterns):
    for patch, pattern in zip(ax.patches, patterns):
        patch.set_hatch(pattern)


with open('data/results/mind2web/runtime/runtime-0.json', 'r') as infile:
    data1 = json.load(infile)
with open('data/results/mind2web/runtime/runtime-1.json', 'r') as infile:
    data2 = json.load(infile)
with open('data/results/mind2web/runtime/runtime-2.json', 'r') as infile:
    data3 = json.load(infile)
with open('data/results/mind2web/runtime/runtime-3.json', 'r') as infile:
    data4 = json.load(infile)

new_data = {'Component': [], 'Times (s)': []}
for key, val in data1.items():
    for i in range(len(val['times'])):
        new_data['Component'].append(key)
    new_data['Times (s)'].extend(val['times'])
for key, val in data2.items():
    for i in range(len(val['times'])):
        new_data['Component'].append(key)
    new_data['Times (s)'].extend(val['times'])
for key, val in data3.items():
    for i in range(len(val['times'])):
        new_data['Component'].append(key)
    new_data['Times (s)'].extend(val['times'])
for key, val in data4.items():
    for i in range(len(val['times'])):
        new_data['Component'].append(key)
    new_data['Times (s)'].extend(val['times'])

df = pd.DataFrame(new_data)

replace_dict = {'process_screenshot': 'Process Screenshot', 'get_page_context': 'Page Context', 'filter_elements': 'Filter Elements', 'limit_elements': 'Limiting Strings', 'element_proposal': 'Candidate Proposal', 'makes_sense_checking': 'Double Checking', 'element_action_selection': 'Element Action Selection', 'end_state': 'End State', 'text_field': 'Text Field', 'select_option': 'Select Option'}


for component in df['Component'].unique():
    print(component)
    print(df['Times (s)'].where(df['Component'] == component).median())

df['Component'] = df['Component'].replace(replace_dict)
plt.figure()
sns.set_theme(style='whitegrid', font_scale=1.2)
# plt.figure(figsize=(12, 6))
g = sns.boxplot(x='Component', y="Times (s)", data=df, showfliers=False, order=['Process Screenshot', 'Page Context', 'Filter Elements', 'Limiting Strings', 'Candidate Proposal', 'Double Checking', 'Element Action Selection', 'Text Field', 'Select Option', 'End State'])
plt.xlabel('Component')
plt.ylabel('Times (s)')
# plt.yscale('log')
# plt.ylabel("Tokens")
plt.title('Component Runtimes')
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
# add_pattern_to_boxes(g, patterns)
plt.savefig('data/plots/component_runtimes.pdf')
exit()

# cost_info = {
#     "FT": (12, 16),
#     "GPT-3.5-Turbo": (1.5, 2),
#     "GPT-3.5-Long": (3, 4),
#     "GPT-4": (30, 60),
#     "Anyscale": (1, 1),
# }

# component_map = {
#     'Candidate Proposal': 'GPT-3.5-Turbo',
#     'Candidate Action Selection': 'GPT-4',
#     'Makes Sense': 'FT',
#     'Secondary Parameter': 'GPT-4',
#     'High Level': 'GPT-3.5-Turbo',
#     'End State': 'FT',
#     'Page Context': 'GPT-3.5-Turbo',
# }

def calculate_cost(row):
    component = row['Component']
    input_tokens = row['Input Tokens (Short)']
    output_tokens = row['Output Tokens']
    
    # Lookup the model from the component map
    model = component_map.get(component)
    
    if model:
        # Lookup the cost info based on the model
        cost_range = cost_info.get(model)
        if cost_range:
            # Calculate cost based on the cost info and tokens
            cost = (cost_range[0] + cost_range[1]) / 2 * (input_tokens + output_tokens) / 1_000_000
            return cost
    
    return None


def plot_category_counts(data, title, category, hue):
    plt.figure()
    sns.set(style='whitegrid', palette=MUTED)
    # plt.figure(figsize=(12, 6))
    sns.barplot(x=category, y="Combined Tokens", data=data, hue=hue, hue_order=['Travel', 'Entertainment', 'Shopping'], order=['Movie', 'Game', 'Music', 'Event', 'Sports', 'Ground', 'Restaurant', 'Other', 'Car rental', 'Airlines', 'Hotel', 'General', 'Department', 'Speciality', 'Fashion', 'Digital'])
    # sns.barplot(x=category, y="Combined Tokens", data=data, hue=hue)
    plt.xlabel(category)
    plt.ylabel("Tokens")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.legend()
    plt.savefig('data/plots/'+title+'.pdf')

def plot_dist(data, title, category):
    data['Component'] = data['Component'].replace(replace_dict)
    order = data.groupby('Component')['Cost'].mean().sort_values().index
    plt.figure()
    sns.set(style='whitegrid', palette=MUTED)
    # plt.figure(figsize=(12, 6))
    g = sns.boxplot(x=category, y="Cost", data=data, showfliers=False, order=order)
    plt.xlabel(category)
    plt.ylabel('Cost ($)')
    # plt.yscale('log')
    g.set(ylim=[0, 0.025])
    # plt.ylabel("Tokens")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    sns.despine()
    plt.savefig('data/plots/'+title+'.pdf')

def plot_cost_per_task_subdomains(df, hue):
    sns.set(style='whitegrid', font_scale=1.25)
    df['Cost'] = df.groupby('Task ID')['Cost'].transform('sum')
    df.drop_duplicates(subset=['Task ID'], inplace=True)
    print(np.mean(df['Cost']))
    for domain in df['Subdomain'].unique():
        print(domain)
        print(np.mean(df.where(df['Subdomain'] == domain)['Cost']))

    order = df.groupby('Subdomain')['Cost'].mean().sort_values().index
    plt.figure()
    g = sns.boxplot(data=df, y='Cost', x='Subdomain', fliersize=2, order=order)
    g.set(ylim=[0, 2.5])
    plt.title('Total Cost to Run a Task with Steward')
    plt.ylabel('Cost ($)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('data/plots/'+'Total Tokens by Subdomain for each Task ID'+'.pdf')

df['Cost'] = df.apply(calculate_cost, axis=1)

plot_dist(df.copy(), "Component Cost Per Action Step", 'Component')

plot_category_counts(df.copy(), "Token Counts by Subdomain", 'Subdomain', 'Domain')

plot_cost_per_task_subdomains(df.copy(), 'Domain')


subdomain_groups = orig_df.groupby('Subdomain')

# Initialize an empty dictionary to store the results
average_actions_per_task = {}

# Iterate through each subdomain group
for subdomain, group_data in subdomain_groups:
    # Calculate the average number of action IDs per task ID
    average_actions = group_data['Task ID'].value_counts().mean()
    # Store the result in the dictionary
    average_actions_per_task[subdomain] = average_actions

# Print the average actions per task for each subdomain
for subdomain, average_actions in average_actions_per_task.items():
    print(f"Subdomain: {subdomain}, Average Actions per Task: {average_actions / 7}")

