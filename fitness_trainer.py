from typing import Deque, Tuple
from pyglet.gl import glClearColor
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
    def __init__(
        self,
        model: ActivityRecognizer,
        sensor: SensorUDP,
        session: TrainingSession,
        window_seconds: float = 1,
        sample_rate: int = 60,
    ):
        super().__init__(Config.window_width, Config.window_height, "Fitness Trainer")
        self.model = model
        self.sensor = sensor
        self.window_size = int(window_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.session = session
        self.current_stage = 0
        self.acc_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=self.window_size)
        self.gyro_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=self.window_size)

        # Activity prediction buffer and threshold
        self.prediction_buffer: Deque[str] = deque(maxlen=self.window_size // 4)

        # Init graphics stuff
        self.batch = Batch()
        self.label_display = pyglet.text.Label(
            "Initializing...",
            font_size=12,
            x=self.width // 2,
            y=0,
            anchor_x="center",
            anchor_y="bottom",
            batch=self.batch,
            color=(24, 24, 24, 255),
        )

        self.stage_display = StageDisplay(self.batch)
        self.stage_display.set_data(session.stages[self.current_stage])  # Set the first stage

        # Initialize session info display
        self.session_info_display = SessionInfoDisplay(self.batch, session)
        self.session_info_display.update(self.current_stage)

        print(f"DIPPID server listening on {get_ip()}:{self.sensor._port}")
        
        # Start the pyglet loop
        glClearColor(24, 24, 24, 1.0)
        pyglet.clock.schedule_interval(self.update, 1.0 / self.sample_rate)
        pyglet.app.run()

    def on_resize(self, width, height):
        Config.window_width = width
        Config.window_height = height
        return super().on_resize(width, height)

    def device_idle(self, window: pd.DataFrame) -> bool:
        """Use distance from idle state to determine if the device is idle."""

        if len(window) < self.window_size:
            return True

        # Extract accelerometer data features
        acc_features = window[["acc_x", "acc_y", "acc_z"]].values
        std_x = np.std(acc_features[:, 0])
        std_y = np.std(acc_features[:, 1])
        std_z = np.std(acc_features[:, 2])

        # Calculate distance from idle state
        distance_from_idle = np.sqrt(std_x**2 + std_y**2 + std_z**2)

        # Threshold for idle detection, lower = less movement
        idle_threshold = 0.3

        return distance_from_idle < idle_threshold

    def is_activity_majority(self, activity_name: str) -> bool:
        if len(self.prediction_buffer) < self.prediction_buffer.maxlen:
            return False

        activity_count = sum(1 for activity in self.prediction_buffer if activity == activity_name)
        activity_ratio = activity_count / len(self.prediction_buffer)

        return activity_ratio >= 0.6

    def update(self, dt: float):
        # Check if the stage is complete and advance to the next one if it is
        if self.stage_display.is_complete():
            self.current_stage += 1
            if self.current_stage >= len(self.session.stages):
                self.label_display.text = "All stages completed!"
                self.stage_display.set_data(None)
                # Update session info for completion
                self.session_info_display.update(self.current_stage)
                return

            self.stage_display.set_data(self.session.stages[self.current_stage])
            # Update session info for new stage
            self.session_info_display.update(self.current_stage)
            return

        # Get the current sensor value and append it to the buffers
        acc = self.sensor.get_value("accelerometer")
        gyro = self.sensor.get_value("gyroscope")

        if acc is None or gyro is None:
            self.label_display.text = "Not connected to sensor!"
            self.label_display.color = (245, 135, 40, 255)
        else:
            self.label_display.text = "No activity detected. Complete all exercises to advance."
            self.label_display.color = (120, 165, 70, 255)
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

        if len(self.acc_buffer) < self.window_size or self.device_idle(window):
            target_y = 0
            current_y = self.label_display.y
            self.label_display.y = current_y + (target_y - current_y) * 0.1
            return  # Not enough data to make a prediction or device is idle

        prediction, confidence = self.model.predict(window)
        confidence_threshold_met = confidence >= 0.99

        # Add the current prediction to the buffer if confidence threshold is met
        if confidence_threshold_met:
            self.prediction_buffer.append(prediction)

        # Check if the current activity maintains a majority in the prediction buffer
        activity_meets_majority = self.is_activity_majority(prediction)

        # Update stage display with prediction only if activity meets majority threshold
        if activity_meets_majority:
            self.stage_display.update(dt, prediction)

        # Slide info text based on confidence level and majority threshold
        target_y = -self.label_display.content_height if activity_meets_majority else 0
        current_y = self.label_display.y
        self.label_display.y = current_y + (target_y - current_y) * 0.1

    def on_draw(self):
        self.clear()
        self.batch.draw()


@click.command()
@click.option("--data-dir", default="data", help="Path to the raw CSV data directory", type=str)
@click.option("--model-output", default="svm_model.pkl", help="Path to save the trained model", type=str)
@click.option("--port", default=5700, help="Port for the UDP sensor", type=int)
@click.option("--session-file", default="sessions/balanced.json", help="Path to the session file", type=str)
def main(data_dir: str, model_output: str, port: int, session_file: str):
    preprocessor = Preprocessor(data_dir)
    processed_data = preprocessor.process_all_files()

    model = ActivityRecognizer(processed_data)
    try:
        model.load(model_path=model_output)
    except Exception:
        print(f"Unable to load model from {model_output}. Training a new model.")
        model.train(model_output_path=model_output)
    model.load(model_path=model_output)

    session = TrainingSession(session_file)
    FitnessTrainer(model, SensorUDP(port), session)


if __name__ == "__main__":
    main()
