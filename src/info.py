from typing import Optional
import pyglet
from src.config import Config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training import TrainingSession, Stage


class SessionInfoDisplay:
    def __init__(self, batch: pyglet.graphics.Batch, session: Optional["TrainingSession"] = None):
        self.batch = batch
        self.session = session
        self.current_stage_index = 0
        self.info_area_height = Config.window_height * 0.3  # Bottom 30% of screen

        # Set up text labels
        self.stage_name_label = pyglet.text.Label(
            "Stage: N/A",
            font_size=20,
            x=Config.window_width // 2,
            y=self.info_area_height - 30,
            anchor_x="center",
            anchor_y="center",
            batch=self.batch,
            color=(24, 24, 24, 255),
        )

        self.progress_label = pyglet.text.Label(
            "Progress: N/A",
            font_size=14,
            x=Config.window_width // 2,
            y=self.info_area_height - 60,
            anchor_x="center",
            anchor_y="center",
            batch=self.batch,
            color=(24, 24, 24, 255),
        )

        self.next_set_label = pyglet.text.Label(
            "Next activities: N/A",
            font_size=14,
            x=Config.window_width // 2,
            y=self.info_area_height - 90,
            anchor_x="center",
            anchor_y="center",
            batch=self.batch,
            color=(24, 24, 24, 255),
            width=Config.window_width - 40,
        )

    def update(self, current_stage_index):
        if self.session is None or current_stage_index >= len(self.session.stages):
            self.stage_name_label.text = "Training Complete!"
            self.progress_label.text = ""
            self.next_set_label.text = ""
            return

        self.current_stage_index = current_stage_index
        current_stage: "Stage" = self.session.stages[current_stage_index]

        # Update stage name and set number
        self.stage_name_label.text = f"Current Set: {current_stage.name}"

        # Update progress
        self.progress_label.text = f"Progress: {current_stage_index + 1} of {len(self.session.stages)} sets"

        # Update next set's activities preview
        if current_stage_index + 1 < len(self.session.stages):
            next_stage: "Stage" = self.session.stages[current_stage_index + 1]
            # List activities in next set
            next_activities = [activity.name for activity in next_stage.activities]
            self.next_set_label.text = f"Upcoming activities: {', '.join(next_activities)}"
        else:
            self.next_set_label.text = ""
