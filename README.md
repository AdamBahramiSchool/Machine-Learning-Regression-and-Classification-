# Machine-Learning-Regression-and-Classification-

# Machine Learning Tasks

This project involves two main machine learning tasks: CPU Temperature Regression and Colour Words Classification.

## CPU Temperature Regression

The goal of this task is to predict the CPU temperature at time t+1 based on historical data. We use a linear regression model from scikit-learn and fit it to the training data. The data set is provided as separate training and validation sets: `sysinfo-train.csv` and `sysinfo-valid.csv`.

To run the program, use the following command:

```bash
    python3 regress_cpu.py sysinfo-train.csv sysinfo-valid.csv

The ‘next_temp’ column in the DataFrame is filled in when it’s read. This represents the temperature at time t+1, which is what we want to predict. The transition matrix for the Kalman filter is updated to use the new-and-improved predictions for temperature.

## Colour Words

In this task, we have collected data mapping colours (specifically RGB colours) to colour words. The data set, colour-data.csv, contains almost 4000 data points. The goal is to train a classifier to predict colour words based on RGB values.

To run the program, use the following command:

python3 colour_bayes.py colour-data.csv

The RGB values are normalized to the 0–1 range and a naïve Bayes classifier is trained on the data. The accuracy score of the model is printed for evaluation.

## Colour Words and Colour Distances

In this part, we convert the RGB colours to LAB colours, which is a more perceptually uniform colour space. A pipeline model is created where the first step is a transformer that converts from RGB to LAB, and the second is a Gaussian classifier, exactly as before. The accuracy value for this model is also evaluated.
