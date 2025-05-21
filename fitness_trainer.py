from typing import Deque, Optional, Tuple
import pandas as pd
import click
import pyglet
from pyglet.window import Window
from pyglet.graphics import Batch
from collections import deque
from activity_recognizer import ActivityRecognizer, Preprocessor
from DIPPID import SensorUDP
from gather_data import get_ip
from src.config import Config
from src.table import StageDisplay
from src.info import SessionInfoDisplay
from src.training import TrainingSession
import numpy as np


class FitnessTrainer(Window):
    _state: str = "idle"
    _current_stage_completed_duration: float = 0.0

    def __init__(
        self,
        model: ActivityRecognizer,
        sensor: SensorUDP,
        session: TrainingSession,
    ):
        super().__init__(Config.window_width, Config.window_height, "Fitness Trainer")
        self.model = model
        self.sensor = sensor
        self.session = session
        self.current_stage = 0
        self.acc_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=Config.LIVE_DATA_SUBSET_SIZE)
        self.gyro_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=Config.LIVE_DATA_SUBSET_SIZE)

        # Activity prediction buffer and threshold
        self.prediction_buffer: Deque[Optional[str]] = deque(maxlen=Config.LIVE_DATA_SUBSET_SIZE // 4)

        # Init graphics stuff
        self.batch = Batch()

        self.background = pyglet.shapes.Rectangle(
            x=0,
            y=0,
            width=self.width,
            height=self.height,
            color=(24, 24, 24, 255),
            batch=self.batch,
        )

        self.stage_display = StageDisplay(self.batch)
        self.stage_display.set_data(session.stages[self.current_stage])  # Set the first stage

        # Initialize session info display
        self.session_info_display = SessionInfoDisplay(self.batch, session, self.stage_display)

        print(f"DIPPID server listening on {get_ip()}:{self.sensor._port}")

        # Start the pyglet loop
        pyglet.clock.schedule_interval(self.update, 1.0 / Config.UPDATE_RATE)
        pyglet.app.run()

    def on_resize(self, width, height):
        Config.window_width = width
        Config.window_height = height
        return super().on_resize(width, height)

    def device_idle(self, window: pd.DataFrame, threshold: float = 0.15) -> Tuple[bool, float]:
        """Use distance from idle state to determine if the device is idle."""

        if len(window) < Config.IDLE_DATA_SUBSET_SIZE:
            return (True, 0.0)  # Not enough data check if idle yet

        acc = window[["acc_x", "acc_y", "acc_z"]].values
        mag_acc = np.linalg.norm(acc, axis=1)
        std_acc = np.std(mag_acc)

        is_idle = std_acc < threshold

        self._state = "idle" if is_idle else "active"

        return self._state == "idle", min(1, std_acc / threshold)

    def is_activity_majority(self, activity_name: str) -> bool:
        if len(self.prediction_buffer) < self.prediction_buffer.maxlen:
            return False

        activity_count = sum(
            1 for activity in self.prediction_buffer if activity == activity_name and activity is not None
        )
        activity_ratio = activity_count / len(self.prediction_buffer)

        return activity_ratio >= 0.5

    def update(self, dt: float):
        self.session_info_display.update(dt)

        if self.stage_display.is_complete():
            self._current_stage_completed_duration += dt

        # Handle stage logic and update both the stage display and the overview/info display if stage transition is done
        if self._current_stage_completed_duration >= Config.STAGE_TRANSITION_DURATION:
            self.current_stage += 1
            if self.current_stage >= len(self.session.stages):
                self.stage_display.set_data(None)
                return

            self.stage_display.set_data(self.session.stages[self.current_stage])
            self._current_stage_completed_duration = 0.0
            return

        # Get the current sensor value and append it to the buffers
        acc = self.sensor.get_value("accelerometer")
        gyro = self.sensor.get_value("gyroscope")

        if acc is not None and gyro is not None:
            self.acc_buffer.append((acc["x"], acc["y"], acc["z"]))
            self.gyro_buffer.append((gyro["x"], gyro["y"], gyro["z"]))

        # Prepare data and make prediction
        data = {
            "acc_x": [x[0] for x in self.acc_buffer],
            "acc_y": [x[1] for x in self.acc_buffer],
            "acc_z": [x[2] for x in self.acc_buffer],
            "gyro_x": [x[0] for x in self.gyro_buffer],
            "gyro_y": [x[1] for x in self.gyro_buffer],
            "gyro_z": [x[2] for x in self.gyro_buffer],
        }
        window = pd.DataFrame(data)
        is_idle, idle_threshold_ratio = self.device_idle(window)
        self.session_info_display.activity_meter.update(dt, idle_threshold_ratio)
        if len(self.acc_buffer) < Config.LIVE_DATA_SUBSET_SIZE or is_idle:
            self.prediction_buffer.append(None)
            return  # Not enough data to make a prediction or device is idle

        prediction, confidence = self.model.predict(window)
        confidence_threshold_met = confidence >= 0.95

        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})                     ", end="\r")

        # Add the current prediction to the buffer if confidence threshold is met
        if confidence_threshold_met:
            self.prediction_buffer.append(prediction)
        else:
            self.prediction_buffer.append(None)

        # Check if the current activity maintains a majority in the prediction buffer
        activity_meets_majority = self.is_activity_majority(prediction)

        # Update stage display with prediction only if activity meets majority threshold
        if activity_meets_majority:
            self.stage_display.update(dt, prediction)

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_close(self):
        self.sensor.disconnect()
        pyglet.app.exit()


@click.command()
@click.option("--data-dir", default="data", help="Path to the raw CSV data directory", type=str)
@click.option("--model-output", default="svm_model.pkl", help="Path to save the trained model", type=str)
@click.option("--port", default=5700, help="Port for the UDP sensor", type=int)
@click.option("--session-file", default="sessions/balanced.json", help="Path to the session file", type=str)
def main(data_dir: str, model_output: str, port: int, session_file: str):
    # Read training data
    preprocessor = Preprocessor(data_dir)
    processed_data = preprocessor.process_all_files()

    # Train the model
    model = ActivityRecognizer(processed_data)
    try:
        model.load(model_path=model_output)
    except Exception:
        print(f"Unable to load model from {model_output}. Training a new model.")
        model.train(model_output_path=model_output)

    # Load model
    model.load(model_path=model_output)
    print("Model loaded successfully")

    # Load session config
    session = TrainingSession(session_file)
    print("Loaded training session successfully")

    print("Loading UI please wait...", end="\r")
    # Start the fitness trainer (pyglet window)
    FitnessTrainer(model, SensorUDP(port), session)


if __name__ == "__main__":
    main()
