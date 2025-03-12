import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating the dataframe from the data
data = {
    'Model': ['GPT-3', 'InstructGPT', 'GPT-3.5', 'GPT-4', 'Llama-3-chat-7B', 
              'GPT-4o', 'DeepSeek', 'Gemini 2.0 Flash', 'avg. Human Result'],
    'Machiavellianism↓': [3.06, 3.32, 3.36, 3.40, 3.22, 3.20, 3.75, 3.83, 4.00, 3.39],
    'Narcissism↓': [3.30, 3.87, 4.03, 4.44, 3.70, 3.96, 4.55, 4.62, 4.81, 3.78],
    'Psychopathy↓': [3.19, 3.41, 3.65, 4.15, 3.65, 4.33, 4.11, 4.44, 4.85, 3.59]
}

df = pd.DataFrame(data)

# Set up the categories (traits) to plot
categories = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
N = len(categories)

# We'll plot a few selected models for clarity
selected_models = ['GPT-3', 'GPT-4', 'GPT-o3-mini', 'avg. Human Result']
colors = ['blue', 'green', 'red', 'purple']

# Angle of each axis in the plot (divide the plot / number of variables)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], categories, size=12)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
plt.ylim(0, 5)

# Plot data
for i, model in enumerate(selected_models):
    model_data = df[df['Model'] == model].iloc[0].loc[categories].values.tolist()
    model_data += model_data[:1]  # Close the loop
    ax.plot(angles, model_data, linewidth=2, linestyle='solid', label=model, color=colors[i])
    ax.fill(angles, model_data, color=colors[i], alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Add title
plt.title('Personality Traits of AI Models', size=15, y=1.1)

plt.tight_layout()
plt.show()

# If you want to plot all models (might be crowded)
def plot_all_models():
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
    plt.ylim(0, 5)
    
    for i, model in enumerate(df['Model']):
        model_data = df[df['Model'] == model].iloc[0].loc[categories].values.tolist()
        model_data += model_data[:1]
        ax.plot(angles, model_data, linewidth=1.5, linestyle='solid', label=model)
        ax.fill(angles, model_data, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Big Five Inventory (BFI)', size=15, y=1.1)
    plt.tight_layout()
    plt.show()
    plt.savefig("BFI.png")

# Uncomment to plot all models
plot_all_models()