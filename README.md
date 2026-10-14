# Student Performance Neural Network Analysis

This project looks at the UCI Student Performance dataset and compares several neural network settings.

## Dataset

The project uses the Math course file from the UCI Student Performance dataset:
- `data/student-mat.csv`

Only the Math CSV is needed to run this project. The other files originally in
the dataset archive included the Portuguese-course data, an R helper script,
and archive/metadata files; this project does not use them. R is not required.

The target is a pass/fail label based on the final grade (`G3`):
- pass = 1 if `G3 >= 10`
- fail = 0 otherwise

## What this code does

- reads the dataset
- cleans and encodes the features
- standardizes the inputs
- trains multiple neural networks using different settings
- records training/test accuracy and loss
- saves a summary table
- plots model history in separate panels for each activation function

The starter template was followed, but the code was expanded to include a clean preprocessing step, model comparison, and output files.

## Main files

- `student_performance_neural_network.py` - training and evaluation code
- `results/model_summary.csv` - table with all model results
- `results/model_history.csv` - accuracy and loss tracked across epochs
- `plots/model_history.png` - test-accuracy history, split into one panel per activation

## How to run

```bash
pip install -r requirements.txt
python student_performance_neural_network.py
```

## Activation function summary

The script compares these activation functions:
- sigmoid
- tanh
- relu

The average test accuracy across the tested settings was:
- tanh: 0.843
- sigmoid: 0.834
- relu: 0.813

So the best overall performer in this setup was `tanh`. The strongest single run was a sigmoid network with `lr=0.10`, `200` epochs, and `3` hidden layers, which reached about `89.9%` test accuracy. Scikit-learn calls the sigmoid activation `logistic`, so the code uses that library-specific name internally while the report and result tables call it sigmoid.

The results are saved in `results/model_summary.csv` and `results/activation_summary.csv`.
