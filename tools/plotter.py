import matplotlib.pyplot as plt
import numpy as np

levels = [1, 2, 3, 4, 5]
model_names = ['Gemma', 'Llama3.2', 'Mistral', 'Phi3', 'Qwen2']

model_colors = {
    'Gemma': 'mediumblue',
    'Llama3.2': 'firebrick',
    'Mistral': 'forestgreen',
    'Phi3': 'darkorange',
    'Qwen2': 'darkorchid'
}

model_styles = {
    'Gemma': ('o', '-'),
    'Llama3.2': ('s', '--'),
    'Mistral': ('^', '-.'),
    'Phi3': ('D', ':'),
    'Qwen2': ('p', '-')
}

textfooler_data = [
    [0.715252, 0.709751, 0.698432, 0.682153, 0.671838],   
    [0.408281, 0.408641, 0.406790, 0.403286, 0.402200],   
    [0.687147, 0.677805, 0.669727, 0.657430, 0.650755],   
    [0.765538, 0.763840, 0.750981, 0.752818, 0.759302],  
    [0.675739, 0.667533, 0.663329, 0.651962, 0.642659]    
]

wordbug_data = [
    [0.723456, 0.730216, 0.715150, 0.701398, 0.720502],   
    [0.408829, 0.407835, 0.408070, 0.405217, 0.407301],  
    [0.685863, 0.687734, 0.684749, 0.679035, 0.679971],   
    [0.764332, 0.758504, 0.757178, 0.756922, 0.758122],  
    [0.675003, 0.682719, 0.679571, 0.679664, 0.672315]    
]

def create_comparison_plot(plot_data, title, filename, y_min=0.40, y_max=0.80):
    plt.figure(figsize=(9, 6))

    for i, model_scores in enumerate(plot_data):
        model = model_names[i]
        marker, linestyle = model_styles[model]
        
        plt.plot(
            levels,
            model_scores,
            label=model,
            color=model_colors[model],
            marker=marker,
            linestyle=linestyle,
            linewidth=2
        )

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Perturbation Level (1: Low Severity to 5: High Severity)', fontsize=11)
    plt.ylabel('Average BERTScore F1', fontsize=11)

    plt.ylim(y_min, y_max)
    plt.xticks(levels)  
    plt.legend(title='Language Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tick_params(axis='x', rotation=15)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(filename)
    print(f"Generated plot: {filename}")


 
create_comparison_plot(
    textfooler_data,
    'Model Robustness to TextFooler Perturbations (BERTScore F1)',
    'textfooler_llm_comparison.png',
    y_min=0.38,   
    y_max=0.78   
)

create_comparison_plot(
    wordbug_data,
    'Model Robustness to WordBug Perturbations (BERTScore F1)',
    'wordbug_llm_comparison.png',
    y_min=0.38,
    y_max=0.78
)