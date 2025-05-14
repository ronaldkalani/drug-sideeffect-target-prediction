# drug-sideeffect-target-prediction
Machine learning pipeline for predicting drug side effects and protein targets using ATC classifications, NLP, and supervised learning.

#  Drug Data Integration and Predictive Modeling for Side Effect and Target Protein Classification in Pharmaceutical Informatics

## Goal  
To build a machine learning pipeline that merges and enriches drug-related data—including drug names, ATC classification, side effects, and protein targets—and applies supervised learning and NLP to predict adverse effects and biological targets.

##  Intended Audience  
- Pharmaceutical data scientists  
- Drug safety researchers  
- Bioinformatics engineers  
- ML/AI developers in healthcare  
- Students in computational biology and health data science

##  Strategy & Pipeline Steps  
1. Load and clean drug data from TSV files  
2. Merge drug identifiers with ATC classifications  
3. Simulate or extract side effects  
4. Encode features and train classifiers for:  
   - Side effect classification  
   - Target protein prediction  
5. Visualize results (bar charts, confusion matrix, network graphs)  
6. Enhance with NLP QA using Hugging Face Transformers  
7. Export the final dataset for reuse  

##  Step 1: Load and Merge Drug Data
```python
import pandas as pd

drug_names = pd.read_csv('/content/drug_names.tsv', sep='\t', header=None)
drug_atc = pd.read_csv('/content/drug_atc.tsv', sep='\t', header=None)

drug_names.columns = ['drug_id', 'drug_name']
drug_atc.columns = ['drug_id', 'atc_code']

merged_df = pd.merge(drug_names, drug_atc, on='drug_id', how='inner')
```

## Step 2: Simulate Side Effects
```python
import numpy as np

np.random.seed(42)
side_effects = ['Nausea', 'Headache', 'None']
merged_df['side_effect'] = np.random.choice(side_effects, size=len(merged_df))
```

##  Step 3: Side Effect Classification
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

X = pd.get_dummies(merged_df[['atc_code']], drop_first=True)
y = LabelEncoder().fit_transform(merged_df['side_effect'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Side Effect Classification Accuracy:", clf.score(X_test, y_test))
```

##  Step 4: Target Protein Prediction (Simulated)
```python
targets = ['COX1', 'COX2', '5HT3', 'H1', 'Beta-Blocker']
merged_df['target_protein'] = np.random.choice(targets, size=len(merged_df))

X = LabelEncoder().fit_transform(merged_df['atc_code']).reshape(-1, 1)
y = LabelEncoder().fit_transform(merged_df['target_protein'])

clf.fit(X, y)
print("Target Prediction Accuracy:", clf.score(X, y))
```

##  Step 5: NLP QA on Excipient Functions
```python
from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = '''
PVP enhances solubility of poorly soluble drugs. HPMC is used for sustained release.
Magnesium stearate is added as a lubricant.
'''

questions = [
    "Which excipient is used for sustained release?",
    "Which one improves solubility?",
    "Which one is a lubricant?"
]

for q in questions:
    print(f"Q: {q}")
    print("A:", qa(question=q, context=context)['answer'])
```

## Step 6: Visualizations

### Side Effect Distribution by ATC Class
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df, x='atc_code', hue='side_effect', palette='Set2')
plt.title("Side Effect Distribution by ATC Class")
plt.xlabel("ATC Code")
plt.ylabel("Drug Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Confusion Matrix for Target Prediction
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

### Network Graph: Drug → Target → Side Effect
```python
import networkx as nx

G = nx.DiGraph()
subset = merged_df[['drug_name', 'target_protein', 'side_effect']].dropna().head(10)

for _, row in subset.iterrows():
    G.add_edge(row['drug_name'], row['target_protein'], label='targets')
    G.add_edge(row['target_protein'], row['side_effect'], label='causes')

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')
plt.title("Drug → Target → Side Effect Network")
plt.show()
```

##  Model Summary

| Task                     | Model           | Accuracy | Notes                                      |
|--------------------------|------------------|----------|---------------------------------------------|
| Side Effect Classification | Random Forest   | ~33%     | Simulated and imbalanced data               |
| Target Protein Prediction  | Random Forest   | 100%     | Simulated data; risk of overfitting         |

##  AGI-Oriented Enhancements
- Use multimodal input: molecular structure, clinical trial data, EHR  
- Integrate biomedical LLMs (BioGPT, PubMedBERT)  
- Build drug-target-pathway knowledge graphs  
- Use continual learning for dynamic safety signal updates  

## Dataset Overview

| File Name              | Description                                |
|------------------------|--------------------------------------------|
| `drug_names.tsv`       | Drug identifiers and names                 |
| `drug_atc.tsv`         | ATC therapeutic classifications            |
| `merged_drug_data.csv` | Final enriched dataset with side effects and targets |

##  References
- Breiman, L. (2001). *Random Forests*. *Machine Learning*, 45(1), 5–32  
- WHO. ATC/DDD Index 2023. https://www.whocc.no/atc_ddd_index/  
- Wang, Z., Clark, N.R., Ma'ayan, A. (2016). *Bioinformatics*, 32(15), 2338–2345  
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. *JMLR*, 12, 2825–2830  
- Vaswani, A. et al. (2017). *Attention is All You Need*  
- Hugging Face Transformers: https://huggingface.co/transformers
