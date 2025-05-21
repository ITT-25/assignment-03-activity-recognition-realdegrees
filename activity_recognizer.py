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


class Preprocessor:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.window_size = Config.TRAINING_DATA_SUBSET_SIZE
        self.step_size = int(Config.TRAINING_DATA_SUBSET_SIZE * 0.8)

    def get_features(self, window: pd.DataFrame) -> dict:
        """
        Extract baseline time-domain features and rotation independent variables
        from accelerometer and gyroscope data in the given window.
        """
        features = {}
        # Time-domain features for each axis
        for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            signal = window[axis].values
            features[f"{axis}_mean"] = np.mean(signal)
            features[f"{axis}_std"] = np.std(signal)
            features[f"{axis}_energy"] = np.sum(signal**2) / len(signal)
            features[f"{axis}_median"] = np.median(signal)

        # Vector magnitudes
        acc_mag = np.sqrt(window["acc_x"] ** 2 + window["acc_y"] ** 2 + window["acc_z"] ** 2)
        gyro_mag = np.sqrt(window["gyro_x"] ** 2 + window["gyro_y"] ** 2 + window["gyro_z"] ** 2)
        features["acc_mag_mean"] = np.mean(acc_mag)  # ! low pca variance
        features["acc_mag_std"] = np.std(acc_mag)  # ! low pca variance
        features["acc_mag_energy"] = np.sum(acc_mag**2) / len(acc_mag)
        features["acc_mag_median"] = np.median(acc_mag)  # ! low pca variance
        features["gyro_mag_mean"] = np.mean(gyro_mag)
        features["gyro_mag_std"] = np.std(gyro_mag)
        features["gyro_mag_energy"] = np.sum(gyro_mag**2) / len(gyro_mag)
        features["gyro_mag_median"] = np.median(gyro_mag)

        # Could add more features like gyro/acc correlation or frequency domain features but results are already pretty good and didn't change much with more features

        # ! This had a negative effect on the model
        # Frequency domain features
        # fft_acc = np.fft.fft(window[["acc_x", "acc_y", "acc_z"]].values, axis=0)
        # fft_gyro = np.fft.fft(window[["gyro_x", "gyro_y", "gyro_z"]].values, axis=0)
        # features["acc_freq_mean"] = np.mean(np.abs(fft_acc))
        # features["gyro_freq_mean"] = np.mean(np.abs(fft_gyro))
        # features["acc_freq_std"] = np.std(np.abs(fft_acc))
        # features["gyro_freq_std"] = np.std(np.abs(fft_gyro))
        # features["acc_freq_energy"] = np.sum(np.abs(fft_acc)**2) / len(fft_acc)
        # features["gyro_freq_energy"] = np.sum(np.abs(fft_gyro)**2) / len(fft_gyro)
        # features["acc_freq_median"] = np.median(np.abs(fft_acc))
        # features["gyro_freq_median"] = np.median(np.abs(fft_gyro))

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

        return pd.DataFrame(data)


class ActivityRecognizer:
    model: Optional[SVC] = None
    encoder: Optional[LabelEncoder] = None
    scaler: Optional[MinMaxScaler] = None

    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data.dropna()
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
            SVC(probability=True, decision_function_shape="ovo", kernel="poly"),
            param_grid={
                "C": [0.1, 0.5, 1],
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
        X = self.scaler.transform(X)
        pred = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        label = self.encoder.inverse_transform(pred)[0]
        return label, probabilities.max()


def run_live_prediction(trainer: ActivityRecognizer, sensor: SensorUDP, window_seconds: float, sample_rate: int):
    window_size = int(window_seconds * sample_rate)
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

            time.sleep(1.0 / sample_rate)

    except KeyboardInterrupt:
        print("\nActivity recognition stopped.")


@click.command()
@click.option("--data-dir", default="data", help="Path to the raw CSV data directory", type=str)
@click.option("--model-output", default="svm_model.pkl", help="Path to save the trained model", type=str)
@click.option("--port", default=5700, help="Port for the UDP sensor", type=int)
@click.option("--window-seconds", default=1, help="Window size in seconds for live prediction", type=float)
@click.option("--sample-rate", default=60, help="Sample rate for live prediction", type=int)
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
