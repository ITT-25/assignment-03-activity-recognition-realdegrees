# this program recognizes activities

import os
import pandas as pd
import numpy as np
from glob import glob
from typing import Deque, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import click
from collections import deque
from DIPPID import SensorUDP
import time
from sklearn.preprocessing import MinMaxScaler
from src.config import Config
import scipy.stats as stats


class Preprocessor:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.window_size = Config.TRAINING_DATA_SUBSET_SIZE
        self.step_size = int(Config.TRAINING_DATA_SUBSET_SIZE * 0.8)

    def get_features(self, window: pd.DataFrame) -> dict:
        """
        Extract extended time-domain, frequency-domain, and statistical features
        from accelerometer and gyroscope data for physical activity classification.
        """
        
        # Ensure the window is not empty and has more than one unique value
        if window.empty or len(window) < 2 or len(window["gyro_x"].unique()) < 2: 
            return {}
        
        features = {}

        # Vector magnitudes
        acc_mag = np.sqrt(window["acc_x"] ** 2 + window["acc_y"] ** 2 + window["acc_z"] ** 2)
        gyro_mag = np.sqrt(window["gyro_x"] ** 2 + window["gyro_y"] ** 2 + window["gyro_z"] ** 2)

        for label, mag in zip(["acc", "gyro"], [acc_mag, gyro_mag]):
            fft_mag = np.abs(np.fft.fft(mag))
            features[f"{label}_mag_mean"] = np.mean(mag)
            features[f"{label}_mag_std"] = np.std(mag)
            features[f"{label}_mag_energy"] = np.sum(mag**2) / len(mag)
            features[f"{label}_mag_median"] = np.median(mag)
            features[f"{label}_mag_min"] = np.min(mag)
            features[f"{label}_mag_max"] = np.max(mag)
            features[f"{label}_mag_iqr"] = stats.iqr(mag)
            features[f"{label}_mag_skew"] = stats.skew(mag)
            features[f"{label}_mag_kurtosis"] = stats.kurtosis(mag)
            features[f"{label}_mag_freq_mean"] = np.mean(fft_mag)
            features[f"{label}_mag_freq_std"] = np.std(fft_mag)
            features[f"{label}_mag_freq_energy"] = np.sum(fft_mag**2) / len(fft_mag)
            features[f"{label}_mag_freq_median"] = np.median(fft_mag)

        # Correlation features
        # Within sensor correlations (acc-acc, gyro-gyro)
        for (a1, a2) in [("acc_x", "acc_y"), ("acc_y", "acc_z"), ("acc_x", "acc_z"),
                ("gyro_x", "gyro_y"), ("gyro_y", "gyro_z"), ("gyro_x", "gyro_z")]:
            features[f"corr_{a1}_{a2}"] = np.corrcoef(window[a1], window[a2])[0, 1]
            
        # Cross-sensor correlations (acc-gyro)
        for acc_axis in ["acc_x", "acc_y", "acc_z"]:
            for gyro_axis in ["gyro_x", "gyro_y", "gyro_z"]:
                features[f"corr_{acc_axis}_{gyro_axis}"] = np.corrcoef(window[acc_axis], window[gyro_axis])[0, 1]
                
        # Correlation between magnitudes
        features["corr_acc_mag_gyro_mag"] = np.corrcoef(acc_mag, gyro_mag)[0, 1]

        return features

    def process_all_files(self):
        all_files = glob(os.path.join(self.raw_data_dir, "*.csv"))
        data = []

        for file in all_files:
            filename = os.path.basename(file)
            activity = filename.split("-")[1]
            try:
                df = pd.read_csv(file)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # Create subsets of each dataset and extract features for each
            for start in range(0, len(df) - self.window_size + 1, self.step_size):
                window = df.iloc[start : start + self.window_size]
                features = self.get_features(window)
                features["label"] = activity
                data.append(features)

        return pd.DataFrame(data).dropna()


class ActivityRecognizer:
    model: Optional[SVC] = None
    encoder: Optional[LabelEncoder] = None
    scaler: Optional[MinMaxScaler] = None

    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.preprocessor = Preprocessor(raw_data_dir=None)

    def train(self, model_output_path="svm_model.pkl"):
        encoder = LabelEncoder()
        y = encoder.fit_transform(self.data["label"])
        X = self.data.drop(columns=["label"])

        # Scale features to [0, 1] range before training
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Use train_test_split to have a final test set for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model using GridSearchCV for hyperparameter tuning
        clf = GridSearchCV(
            SVC(probability=True, decision_function_shape="ovo"),
            param_grid={
                "kernel": ["linear", "poly", "rbf"],
                "C": [0.1, 0.5, 1, 5],
                "gamma": [0.1, 0.5, 1],
                "degree": [2, 3, 4],
                "class_weight": ["balanced", None],
            },
            scoring="accuracy",
            cv=5,
            verbose=10,
            n_jobs=joblib.cpu_count() // 2,
        )

        clf.fit(X_train, y_train)

        # Evaluate on the test set with the best model
        y_pred = clf.predict(X_test)
        print("Best parameters found: ", clf.best_params_)
        print("Best cross-validation score: ", clf.best_score_)
        print("Classification Report:")
        print(classification_report(encoder.inverse_transform(y_test), encoder.inverse_transform(y_pred)))
        print("Accuracy:", accuracy_score(encoder.inverse_transform(y_test), encoder.inverse_transform(y_pred)))

        # Save the best model
        joblib.dump((clf, encoder, scaler), model_output_path)
        print(f"Tuple of (clf, encoder, scaler) saved to {model_output_path}")

    def load(self, model_path="svm_model.pkl"):
        self.model, self.encoder, self.scaler = joblib.load(model_path)

    def predict(self, window: pd.DataFrame) -> Tuple[str, float]:
        features = self.preprocessor.get_features(window)
        X = pd.DataFrame([features])
        X = X.dropna()
        if X.empty:
            return "Unknown", 0.0
        X = self.scaler.transform(X)
        pred = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        label = self.encoder.inverse_transform(pred)[0]
        return label, probabilities.max()


def run_live_prediction(trainer: ActivityRecognizer, sensor: SensorUDP):
    window_size = Config.LIVE_DATA_SUBSET_SIZE
    acc_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=window_size)
    gyro_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=window_size)

    print("Starting activity recognition...")

    try:
        while True:
            acc = sensor.get_value("accelerometer")
            gyro = sensor.get_value("gyroscope")

            if acc is None or gyro is None:
                print("Waiting for sensor data...", end="\r")
                continue

            acc_buffer.append((acc["x"], acc["y"], acc["z"]))
            gyro_buffer.append((gyro["x"], gyro["y"], gyro["z"]))

            if len(acc_buffer) == window_size:
                data = {
                    "acc_x": [x[0] for x in acc_buffer],
                    "acc_y": [x[1] for x in acc_buffer],
                    "acc_z": [x[2] for x in acc_buffer],
                    "gyro_x": [x[0] for x in gyro_buffer],
                    "gyro_y": [x[1] for x in gyro_buffer],
                    "gyro_z": [x[2] for x in gyro_buffer],
                }
                df_window = pd.DataFrame(data)
                prediction = trainer.predict(df_window)
                print(f"Prediction: {prediction}                               ", end="\r")

            time.sleep(1.0 / Config.UPDATE_RATE)

    except KeyboardInterrupt:
        print("\nActivity recognition stopped.")


@click.command()
@click.option("--data-dir", default="data", help="Path to the raw CSV data directory", type=str)
@click.option("--model-output", default="svm_model.pkl", help="Path to save the trained model", type=str)
@click.option("--port", default=5700, help="Port for the UDP sensor", type=int)
def main(data_dir: str, model_output: str, port: int, window_seconds: float, sample_rate: int):
    preprocessor = Preprocessor(data_dir)
    processed_data = preprocessor.process_all_files()

    trainer = ActivityRecognizer(processed_data)
    try:
        trainer.load(model_path=model_output)
    except Exception:
        print(f"Unable to load model from {model_output}. Training a new model.")
        trainer.train(model_output_path=model_output)

    trainer.load(model_path=model_output)

    run_live_prediction(trainer, sensor=SensorUDP(port=port), window_seconds=window_seconds, sample_rate=sample_rate)


if __name__ == "__main__":
    main()
