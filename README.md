# otero_borya_body_mass
Scripts for body mass in biped dinosaurs
# Body mass prediction in theropod dinosaurs using supervised learning

Code and data accompanying Otero & Borya (in review), *Publicación Electrónica
de la Asociación Paleontológica Argentina*.

## Contents
- `notebooks/eda_unsupervised.ipynb` – exploratory data analysis and K-means clustering (Figs. 6–9)
- `notebooks/rfdh_code.ipynb` – random forest, default hyperparameters (RFdh)
- `notebooks/rfoh_code.ipynb` – random forest, optimized hyperparameters (RFoh), grid search and sensitivity analysis (Fig. 10)
- `notebooks/nn_code.ipynb` – neural networks, learning curves and SHAP
- `data/` – training, validation and benchmark datasets (see Tables S1–S6)

## Requirements
Python 3.12.11. Install dependencies with:

    pip install -r requirements.txt

Random seeds are fixed at 42 (data partitioning, random forests, grid search, neural networks).

## Citation
Otero, A., & Borya, A. (in review). [Title]. Publicación Electrónica de la
Asociación Paleontológica Argentina. Please also cite the archived code: [Zenodo DOI].

## License
Code: MIT. Data: CC BY 4.0 (data compiled from published sources; see Table S1 for original references).
