# UnexpectedRelevantUserSimulation
Simulation of a User applied to MI dialog with generation of unexpected yet relevant dialog acts

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main_{condition}.py

With condition being one of the following:
- 1: Full_Model : The full model with all the components
- 2: ablation_OT : Ablation study with the dialog acts component removed
- 3: ablation_ODA : Ablation study with the text component removed
```

## Metrics
The metrics are computed in the `compute_metrics.ipynb` notebook. The metrics are:
- F1 score
- Diversity
- Unexpectedness
- Automatic Relevance
- Serendipity