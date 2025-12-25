import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the results
df = pd.read_csv('rag_evaluation_results.csv')

# Define a function to determine success
def is_successful(answer):
    if not isinstance(answer, str):
        return False
    if answer.strip().startswith("I cannot answer"):
        return False
    if answer.strip().startswith("Error generating answer"):
        return False
    return True

df['Success'] = df['Answer'].apply(is_successful)

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Set the style
sns.set_theme(style="whitegrid")

# 1. Success Rate by Retrieval Method
plt.figure(figsize=(8, 6))
success_by_retrieval = df.groupby('Retrieval Method')['Success'].mean().reset_index()
sns.barplot(x='Retrieval Method', y='Success', data=success_by_retrieval, palette='viridis')
plt.title('Success Rate by Retrieval Method')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('images/success_by_retrieval.png')
plt.close()

# 2. Success Rate by Chunking Method
plt.figure(figsize=(8, 6))
success_by_chunking = df.groupby('Chunking Method')['Success'].mean().reset_index()
sns.barplot(x='Chunking Method', y='Success', data=success_by_chunking, palette='magma')
plt.title('Success Rate by Chunking Method')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('images/success_by_chunking.png')
plt.close()

# 3. Success Rate by K
plt.figure(figsize=(8, 6))
success_by_k = df.groupby('k')['Success'].mean().reset_index()
sns.barplot(x='k', y='Success', data=success_by_k, palette='cubehelix')
plt.title('Success Rate by K (Number of Retrieved Chunks)')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('images/success_by_k.png')
plt.close()

# 4. Success Rate by Question Type
plt.figure(figsize=(8, 6))
success_by_type = df.groupby('Question Type')['Success'].mean().reset_index()
sns.barplot(x='Question Type', y='Success', data=success_by_type, palette='coolwarm')
plt.title('Success Rate by Question Type')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('images/success_by_type.png')
plt.close()

print("Plots generated successfully in 'images/' directory.")
