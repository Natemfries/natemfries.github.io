### AOS C204 Final Project

This repository contains the full code, report, and figures for my project in AOS C204: Introduction to Machine Learning for Physical Sciences. 
The goal of the project was to explore whether globally averaged near-surface atmospheric temperature could be predicted from other atmospheric state variables using simple machine-learning models.

### Report

The pdf of the final report is available [here](assets/AOS%20C204%20Final%20Project.pdf).

### Code

All model training, evaluation, and figure generation are contained in: [Code](/assets/project.py). 
The script loads the processed ECCO dataset, performs exploratory analysis, trains linear, ridge, and SVR models, constructs a phase-folded seasonal regression model, and generates all figures used in the report.

### Figures
All figures generated in the code and included in the final report are available [here](assets/).

### Data

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/daveyw164/jpl-ocean-weather?resource=download&select=full_combined_processed.nc) and is discussed further in the report.
