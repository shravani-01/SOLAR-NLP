import matplotlib.pyplot as plt

models = ['GPT-4o\n(CoT)', 'DeepSeek\n(CoT)',
          'DeepSeek\n(RAG)', 'GPT-4o\n(RAG)',
          'SOLAR\n(ours)']
scores = [0.480, 0.742, 0.697, 0.788, 0.858]
colors = ['#c7dcef', '#c7dcef', '#c7dcef', '#c7dcef', '#1f77b4']

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=0.6)

ax.set_ylim(0, 1.0)
ax.set_ylabel('Macro F1', fontsize=11)
ax.set_title('IR Extraction on OpsBench (Transit, n=211)', fontsize=12)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.015,
        f'{score:.3f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold' if score == max(scores) else 'normal'
    )

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/figure2_macro_f1.pdf', dpi=300, bbox_inches='tight')
plt.show()