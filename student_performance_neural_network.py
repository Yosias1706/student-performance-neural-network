import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATASET_PATH = Path(__file__).resolve().parent / "data" / "student-mat.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class NeuralNet:
    def __init__(self, dataFile, target_column="G3", pass_threshold=10, random_state=42):
        self.raw_input = pd.read_csv(dataFile, sep=";")
        self.target_column = target_column
        self.pass_threshold = pass_threshold
        self.random_state = random_state

    def preprocess(self):
        df = self.raw_input.copy()
        df["pass"] = (df[self.target_column] >= self.pass_threshold).astype(int)

        X = df.drop(columns=[self.target_column, "pass"])
        y = df["pass"]

        X = pd.get_dummies(X, drop_first=False)
        X = X.astype(float)

        self.X = X
        self.y = y
        self.processed_data = X.copy()
        self.processed_data["target"] = y.values

        return self.processed_data

    def train_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        activations = ["logistic", "tanh", "relu"]
        learning_rates = [0.01, 0.1]
        max_iterations = [100, 200]
        num_hidden_layers = [2, 3]

        summary_rows = []
        history_rows = []

        PLOTS_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)

        for activation, learning_rate, epochs, hidden_layers in itertools.product(
            activations, learning_rates, max_iterations, num_hidden_layers
        ):
            hidden_size = 32
            layer_tuple = (hidden_size,) * hidden_layers

            model = MLPClassifier(
                hidden_layer_sizes=layer_tuple,
                activation=activation,
                learning_rate_init=learning_rate,
                max_iter=1,
                warm_start=True,
                random_state=self.random_state,
                solver="adam",
            )

            epoch_history = []

            for epoch in range(1, epochs + 1):
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                train_loss = model.loss_
                test_loss = log_loss(y_test, model.predict_proba(X_test_scaled), labels=[0, 1])

                epoch_history.append(
                    {
                        "activation": activation,
                        "learning_rate": learning_rate,
                        "epochs": epoch,
                        "hidden_layers": hidden_layers,
                        "train_accuracy": train_acc,
                        "test_accuracy": test_acc,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }
                )

            final_train_pred = model.predict(X_train_scaled)
            final_test_pred = model.predict(X_test_scaled)

            summary_row = {
                "activation": activation,
                "learning_rate": learning_rate,
                "max_iterations": epochs,
                "num_hidden_layers": hidden_layers,
                "hidden_layer_size": hidden_size,
                "train_accuracy": accuracy_score(y_train, final_train_pred),
                "test_accuracy": accuracy_score(y_test, final_test_pred),
                "train_loss": model.loss_,
                "test_loss": log_loss(y_test, model.predict_proba(X_test_scaled), labels=[0, 1]),
            }
            summary_rows.append(summary_row)
            history_rows.extend(epoch_history)

        summary_df = pd.DataFrame(summary_rows)
        history_df = pd.DataFrame(history_rows)

        activation_summary = (
            summary_df.groupby("activation")
            .agg(
                runs=("activation", "count"),
                avg_train_accuracy=("train_accuracy", "mean"),
                avg_test_accuracy=("test_accuracy", "mean"),
                avg_train_loss=("train_loss", "mean"),
                avg_test_loss=("test_loss", "mean"),
            )
            .reset_index()
            .sort_values("avg_test_accuracy", ascending=False)
        )

        summary_df.to_csv(RESULTS_DIR / "model_summary.csv", index=False)
        history_df.to_csv(RESULTS_DIR / "model_history.csv", index=False)
        activation_summary.to_csv(RESULTS_DIR / "activation_summary.csv", index=False)

        self.summary_df = summary_df
        self.history_df = history_df
        self.activation_summary = activation_summary

        self._plot_history(history_df)
        return summary_df, history_df, activation_summary

    def _plot_history(self, history_df):
        plt.figure(figsize=(15, 9))
        for activation in ["logistic", "tanh", "relu"]:
            for learning_rate in [0.01, 0.1]:
                for hidden_layers in [2, 3]:
                    subset = history_df[
                        (history_df["activation"] == activation)
                        & (history_df["learning_rate"] == learning_rate)
                        & (history_df["hidden_layers"] == hidden_layers)
                    ]
                    if subset.empty:
                        continue
                    label = (
                        f"{activation} | lr={learning_rate} | layers={hidden_layers}"
                    )
                    plt.plot(subset["epochs"], subset["test_accuracy"], label=label)

        plt.title("Neural network accuracy over training epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_history.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    neural_network = NeuralNet(str(DATASET_PATH))
    neural_network.preprocess()
    summary_df, history_df, activation_summary = neural_network.train_evaluate()

    print("\nBest test accuracy by activation:")
    print(activation_summary[["activation", "avg_test_accuracy"]].to_string(index=False))
    print("\nTop 5 runs:")
    print(summary_df.sort_values("test_accuracy", ascending=False).head(5).to_string(index=False))
