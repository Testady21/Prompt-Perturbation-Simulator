import matplotlib.pyplot as plt
import numpy as np

perturbations = [
    'Casing', 'Homophone', 'Punctuation', 'Paraphrase', 
    'Shuffling', 'Stopword', 'TextFooler L5', 'Typo Swap', 
    'WordBug L5', 'wWordBug_Textfooler_Combined'
]

data = {
    'Gemma': [0.7172, 0.7268, 0.7250, 0.6410, 0.7024, 0.7254, 0.6796, 0.7121, 0.7077, 0.6478],
    'Llama 3.2': [0.6825, 0.6767, 0.6836, 0.5996, 0.6634, 0.6769, 0.6411, 0.6703, 0.6755, 0.6120],
    'Mistral': [0.6809, 0.6916, 0.6914, 0.6262, 0.6622, 0.6807, 0.6503, 0.6760, 0.6720, 0.6365],
    'Phi-3': [0.7749, 0.7816, 0.7731, 0.7739, 0.7761, 0.7846, 0.7744, 0.7746, 0.7693, 0.7706],
    'Qwen-2': [0.6747, 0.6755, 0.6777, 0.6385, 0.6585, 0.6729, 0.6559, 0.6691, 0.6750, 0.6416]
}

x = np.arange(len(perturbations))  
width = 0.15  
 
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
models = list(data.keys())

fig, ax = plt.subplots(figsize=(14, 7))

 
for i, model in enumerate(models):
    offset = (i - len(models)/2) * width + width/2
    ax.bar(x + offset, data[model], width, label=model, color=colors[i], edgecolor='white', linewidth=0.5)

 
ax.set_ylabel('Average BERTScore F1', fontsize=12, fontweight='bold')
ax.set_title('LLM Robustness Profile Across Perturbation Variants', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(perturbations, rotation=25, ha='right', fontsize=10)
 
ax.set_ylim(0.55, 0.85) 
ax.grid(axis='y', linestyle='--', alpha=0.7)

 
ax.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True)

plt.tight_layout()

 
plt.savefig('perturbation_variants_comparison.png', dpi=300)
print("Plot saved as 'perturbation_variants_comparison.png'")