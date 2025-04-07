# Laboration 1 - Medical Predictions

## Introduction

This laboration is part of a machine learning course at [ITHS](www.iths.se), taught by [Raphael Korsoski](www.github.com/pr0fez). The lab focuses on the implementation of various machine learning techniques to classify patients with cardiovascular disease ([link to dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)). 

Five machine learning models are used with a predefined set of hyperparameters and values. These are tuned using grid search cross validation and the best parameters are saved for use in a voting classifier.

## Script

The script file contains three classes:

- Processing
- Visualisation
- Modelling

These do the heavy lifting in the lab to provide pre-processing to the dataset, visualisations for EDA and comparisons and different methods for modelling and evaluation. 

## Report

The report is a short documentation of the lab findings. The idea is to loosely hold true to conventional report structures used by data scientists. 

It imports the classes from the script so that the mainstay of the code is kept separate from the report. 