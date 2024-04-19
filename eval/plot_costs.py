import os, csv, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch


def add_pattern_to_boxes(ax, patterns):
    for patch, pattern in zip(ax.artists, patterns):
        patch.set_hatch(pattern)


df = pd.read_csv('data/costs.csv')    

print(df)

plt.figure()
sns.set_theme(style='whitegrid', font_scale=1.2)
# plt.figure(figsize=(12, 6))
g = sns.lineplot(data=df, x='Category', y='Cost ($)', color="blue", marker="o", label='Cost ($)')
plt.ylabel('Cost ($)')
plt.xlabel('Website Category')
plt.xticks(rotation=90)
ax2 = plt.twinx()
g = sns.lineplot(data=df, x='Category', y='# Actions', color="red", marker="x", ax=ax2, label='Number of Actions')
ax2.set_ylabel('# Actions')
# plt.yscale('log')
# plt.ylabel("Tokens")
plt.title('Task Costs')
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig('data/plots/task_costs.pdf')
exit()

