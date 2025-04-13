import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define evaluation metrics
metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']

# Define performance for each model
data = {
    'Logistic Regression': [0.78, 0.77, 0.77, 0.82],
    'SVM': [0.77, 0.76, 0.76, 0.81],
    'Neural Network': [0.81, 0.81, 0.81, 0.84],
    'Prompting (Zero-Shot)': [0.64, 0.55, 0.32, 0.35]
}

# Convert to DataFrame
df = pd.DataFrame(data, index=metrics).reset_index().melt(id_vars='index')
df.columns = ['Evaluation Metric', 'Model', 'Score']

# Set Seaborn style
sns.set(style='whitegrid', font_scale=1.2)

# Create bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Evaluation Metric', y='Score', hue='Model', data=df)

# Add grid
ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

# Title and labels
plt.title('Comparison of Depression Classification Performance')
plt.ylim(0, 1.0)
plt.ylabel('Score')
plt.xlabel('Evaluation Metric')
plt.legend(title='Model')

# Tight layout for saving
# plt.tight_layout()

# Save figure for use in research paper
plt.savefig('depression_classification_comparison.png', dpi=300)

plt.show()
