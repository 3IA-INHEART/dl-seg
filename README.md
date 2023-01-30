dl-seg-med
==============================

Data
-----
[Zenodo link](https://zenodo.org/record/6802614)

Research question
------------------
We have a dataset D_0 containing 100 CT scans annotated partially with ten classes. On D_0, we train a (baseline) model M_0 (using only supervised learning).

We also have a dataset D_1, containing another 100 CT scans, partially annotated with 20 classes (including those from the D_0).

Remark. We will simulate D_0 and D_1 by subsampling data from the [open dataset](https://zenodo.org/record/6802614) and changing some labels to 0 (background).

Question. Can we build a model M_1 based on combined data (D_0 + D_1) using a semi-supervised approach so that the model M_1 performs better than M_0 (at least for some classes)?
Additional question. If D_1 is larger or smaller than D_0, how will it affect the performance gain of M_1?

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── dl-seg-med           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── validate                            <- Scripts to validate models
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations
