dl-seg-med
==============================

Data
-----
[Zenodo link](https://zenodo.org/record/6802614)

Research question
------------------
We have a dataset D_0 containing 100 CT scans fully annotated with 10 classes. On this dataset, some U-net like model M_0 using supervised learning approach is trained and a particular DICE quality metric is observed on a D_val validation dataset (fully annotated too).
We also have a dataset D_1, containing another 100 CT scans, annotated partially by the maximum of the same 10 classes as the D_0.
Remark. We will simulate D_0, D_1, and D_val by subsampling data from the [open dataset](https://zenodo.org/record/6802614) and hiding some labels for D_1.

Question. Can we build a model M_1 based on combined data (D_0 + D_1) using semi-supervised approach, so that the model M_1 will show better performance (in terms of DICE metric) than M_0 on D_val.
Additinal question. If D_1 has larger or smaller, how it will affect the perofrmance gain of M_1

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
