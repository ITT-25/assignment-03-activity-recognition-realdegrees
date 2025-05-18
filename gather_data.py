import sys
from DIPPID import SensorUDP
import click
import time
import threading
import pandas as pd
import socket
import numpy as np

BAR_LENGTH = 30  # Length of the progress bar


def get_ip():
    """Get the local IP address of the machine using a temporary socket."""

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
    s.close()
    return local_ip


class CaptureData:
    def __init__(self, activity: str, prefix: str, duration: float, sets: int, set_offset: int, delay: float, port: int, sampling_rate: int = 100):
        self.delay = delay
        self.activity = activity
        self.prefix = prefix
        self.duration = duration
        self.sets = sets
        self.set_offset = set_offset
        self.current_set = 0
        self.sensor = SensorUDP(port)
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate

        print(f"DIPPID server listening on {get_ip()}:{port}")
        self.df = pd.DataFrame(columns=["id", "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        self.collecting_data = threading.Event()

    def start(self):
        self._prompt_next_set()

    def _prompt_next_set(self):
        """Prompt the user for the next set of data collection."""

        def start_set(pressed):
            if pressed == 0:
                return

            self.sensor.unregister_callback("button_1", start_set)
            self.sensor.unregister_callback("button_2", redo_set)

            if len(self.df) > 0:
                self.df.to_csv(f"data/{self.prefix}-{self.activity}-{self.set_offset + self.current_set}.csv", index=False)
                self.df.drop(self.df.index, inplace=True)

            if self.current_set >= self.sets:
                print("[All Sets Complete] Data saved. Exiting...")
                sys.exit(0)

            for i in np.arange(self.delay, 0, -0.1):
                print(f"Starting in {i:.1f} seconds...", end="\r")
                time.sleep(0.1)

            self._capture_set()

        def redo_set(pressed):
            if pressed == 0:
                return

            self.sensor.unregister_callback("button_1", start_set)
            self.sensor.unregister_callback("button_2", redo_set)

            self.current_set = max(0, self.current_set - 1)
            self.df.drop(self.df.index, inplace=True)

            self._prompt_next_set()

        self.sensor.register_callback("button_1", start_set)
        self.sensor.register_callback("button_2", redo_set)

        if self.current_set < self.sets:
            print(f"[Set: {self.current_set + 1}/{self.sets}] Button 1: Save and capture next set, Button 2: Redo set")
        else:
            print("[All Sets Complete] Button 1: Save and exit, Button 2: Redo last set")

    def _capture_set(self):
        """Start data collection and progress display for a single set in separate threads."""

        self.collecting_data = threading.Event()

        data_thread = threading.Thread(target=self._collect_sensor_data)
        data_thread.daemon = True
        data_thread.start()

        progress_thread = threading.Thread(target=self._display_progress)
        progress_thread.daemon = True
        progress_thread.start()

    def _collect_sensor_data(self):
        """Collect sensor data (acc and gyro) for the specified duration."""

        start_time = time.time()

        while not self.collecting_data.is_set():
            if time.time() - start_time >= self.duration:
                break

            acc_data = self.sensor.get_value("accelerometer")
            gyro_data = self.sensor.get_value("gyroscope")

            if acc_data is not None and gyro_data is not None:
                self.df.loc[len(self.df)] = {
                    "id": len(self.df),
                    "timestamp": time.time(),
                    "acc_x": acc_data["x"],
                    "acc_y": acc_data["y"],
                    "acc_z": acc_data["z"],
                    "gyro_x": gyro_data["x"],
                    "gyro_y": gyro_data["y"],
                    "gyro_z": gyro_data["z"],
                }

            elapsed = time.time() - start_time
            next_sample_time = start_time + (int(elapsed / self.sampling_interval) + 1) * self.sampling_interval
            sleep_time = max(0, next_sample_time - time.time())
            time.sleep(sleep_time)

    def _display_progress(self):
        """Display a progress bar for the data collection. Increment set when progress is complete and start next prompt."""

        start_time = time.time()
        while not self.collecting_data.is_set():
            elapsed = time.time() - start_time
            if elapsed > self.duration:
                self.collecting_data.set()
                break

            progress = min(elapsed / self.duration, 1.0)
            filled_length = int(BAR_LENGTH * progress)
            bar = "█" * filled_length + "░" * (BAR_LENGTH - filled_length)
            percent = progress * 100

            print(
                f"[Set: {self.current_set + 1}/{self.sets}] Progress: [{bar}] {percent:.1f}% ({len(self.df)} rows) ({self.duration - elapsed:.1f}s left)",
                end="\r",
            )
            time.sleep(0.25)

        print(
            f"[Set: {self.current_set + 1}/{self.sets}] Progress: [{'█' * BAR_LENGTH}] 100% ({len(self.df)} rows)                             "
        )
        self.current_set += 1
        self._prompt_next_set()


@click.command()
@click.option("--activity", "-a", help="Activity name being captured", type=str, default="activity")
@click.option("--prefix", "-p", help="Prefix for CSV filename", type=str, default="prefix")
@click.option("--duration", "-d", help="Duration in seconds", type=float, default=10)
@click.option("--sets", "-s", help="Number of sets to capture", type=int, default=5)
@click.option("--set-offset", "-o", help="Offset the start number for captured sets (Useful if you already have existing data sets)", type=int, default=0)
@click.option("--delay", help="Delay before capture in seconds", type=float, default=3)
@click.option("--port", help="DIPPID client port", type=int, default=5700)
@click.option("--sampling-rate", help="Sampling rate in Hz", type=int, default=200)
def run(activity, prefix, duration, sets, set_offset, delay, port, sampling_rate):
    CaptureData(activity, prefix, duration, sets, set_offset, delay, port, sampling_rate).start()


if __name__ == "__main__":
    run()
