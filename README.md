# Student Performance Neural Network

This project uses the UCI Student Performance dataset to compare neural networks with different activation functions and settings.

## Dataset

The project uses the Math course data hosted in this repository at `data/student-mat.csv`. It predicts whether a student passes based on the final grade (`G3`): grades of 10 or higher are labeled as a pass.

## Run

Install the required packages and run the script:

```bash
pip install -r requirements.txt
python student_performance_neural_network.py
```

The script writes model results to `results/` and the training history plot to `plots/model_history.png`.
