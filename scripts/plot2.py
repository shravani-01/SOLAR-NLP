import matplotlib.pyplot as plt
import numpy as np

categories = [
    'Extraction\n(span / IR)',
    'Conditional\nRealization',
    'Semantic\nGrounding\n(strict)'
]

gpt4o = [95.3, 0.0, 0.0]
solar = [26.2, 94.8, 20.0]

x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(figsize=(7.6, 4.6))

bars1 = ax.bar(
    x - width/2, gpt4o, width,
    label='GPT-4o (CoT)',
    color='#c7dcef',
    edgecolor='black',
    linewidth=0.6
)
bars2 = ax.bar(
    x + width/2, solar, width,
    label='SOLAR (ours)',
    color='#1f77b4',
    edgecolor='black',
    linewidth=0.6
)

ax.set_ylabel('Accuracy / Rate (%)', fontsize=11)
ax.set_title('Exception Handling Gap: Extraction vs. Executable Grounding', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 110)
ax.legend(fontsize=9, frameon=False)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 2,
            f'{h:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax.text(
    x[0] + width/2,
    10,
    'strict\nIR match',
    ha='center',
    va='bottom',
    fontsize=8,
    color='black'
)

ax.text(
    x[1] - width/2,
    8,
    'no executable\nconstraint output',
    ha='center',
    va='bottom',
    fontsize=8,
    color='dimgray'
)

ax.text(
    x[2] - width/2,
    8,
    'not defined',
    ha='center',
    va='bottom',
    fontsize=8,
    color='dimgray'
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/figure3_exception_gap.pdf', dpi=300, bbox_inches='tight')
plt.show()