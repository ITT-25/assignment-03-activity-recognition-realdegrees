from typing import List, Optional
import pyglet
from src.config import Config
from src.training import Activity, Stage
from src.util import load_activity_images

COLUMN_PADDING = 20
ACTIVITY_TYPES = [
    "jumpingjack",
    "running",
    "lifting",
    "rowing",
]  # TODO: Should be dynamically read from model labels so new activities can be added without code changes


class ActivityDisplay:
    def __init__(
        self,
        batch: pyglet.graphics.Batch,
        activity: Activity = None,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
    ):
        self.batch = batch
        self.activity = activity
        self.completion = 0.0
        self.last_image_update_delta = 0.0
        self.x, self.y, self.width, self.height = x, y, width, height

        # Create background rect
        rect_x = self.x - self.width / 2
        self.background = pyglet.shapes.Rectangle(
            x=rect_x,
            y=self.y - self.height,
            width=self.width,
            height=self.height,
            color=Config.PRIMARY_COLOR,
            batch=self.batch,
        )

        # Don't setup display elements if this is just a placeholder display
        if activity is None:
            return

        # Setup display elements
        self._setup_display_elements(rect_x)

    def _setup_display_elements(self, rect_x):
        margin = self.width * 0.05
        font_size = max(10, min(24, int(self.width * 0.07)))
        offset = 20

        activity_name_map = {
            "jumpingjack": "Jumping Jacks",
            "running": "Running",
            "lifting": "Lifting",
            "rowing": "Rowing",
        }
        # Create activity name label
        self.label = pyglet.text.Label(
            activity_name_map.get(self.activity.name, self.activity.name),
            font_size=font_size,
            x=self.x,
            y=self.y - offset,
            anchor_x="center",
            anchor_y="center",
            batch=self.batch,
            color=Config.TEXT_COLOR,
        )
        offset += margin + self.label.content_height

        # Load activity images
        self.images = load_activity_images(self.activity.name)

        # Calculate available space for image
        progress_bar_height = 15
        progress_label_height = font_size + margin

        # Create and scale image sprite
        self.image_sprites = [
            pyglet.sprite.Sprite(
                self.images[i], x=self.x, y=self.y - offset, batch=self.batch)
            for i in range(len(self.images))
        ]
        for i, sprite in enumerate(self.image_sprites):
            sprite.scale = (self.width - margin * 2) / sprite.width
            sprite.x = self.x - sprite.width / 2
            if i != 0:
                sprite.visible = False  # Hide all but the first image

        offset += margin + self.image_sprites[0].image.height * self.image_sprites[0].scale

        # Progress label
        # Format duration with appropriate time units
        if self.activity.duration < 60:
            duration_text = f"For {self.activity.duration} seconds"
        elif self.activity.duration < 3600:
            minutes = int(self.activity.duration // 60)
            seconds = int(self.activity.duration % 60)
            duration_text = f"For {minutes}:{seconds:02d} minutes"
        else:
            hours = int(self.activity.duration // 3600)
            minutes = int((self.activity.duration % 3600) // 60)
            duration_text = f"For {hours}:{minutes:02d} hours"

        self.duration_label = pyglet.text.Label(
            duration_text,
            font_size=font_size,
            x=self.x + self.width // 2 - margin,
            y=self.y - self.height + progress_label_height + margin,
            anchor_x="right",
            anchor_y="center",
            batch=self.batch,
            color=Config.TEXT_COLOR,
        )

        # Progress shapes
        progress_bar_y = self.y - self.height
        self.progress_bg = pyglet.shapes.Rectangle(
            x=rect_x,
            y=progress_bar_y,
            width=self.width,
            height=progress_bar_height,
            color=Config.SECONDARY_COLOR,
            batch=self.batch,
        )

        self.progress_bar = pyglet.shapes.Rectangle(
            x=rect_x,
            y=progress_bar_y,
            width=0,
            height=progress_bar_height,
            color=Config.SUCCESS_COLOR,
            batch=self.batch,
        )

    def update(self, dt: float):
        if self.activity is None:
            return

        # Update completion and set complete flag state
        self.completion = min(self.completion + dt, self.activity.duration)
        is_complete = self.completion >= self.activity.duration

        # Handle image animation
        self.last_image_update_delta += dt
        if self.last_image_update_delta >= Config.IMAGE_ANIMATION_INTERVAL and not is_complete:
            self.last_image_update_delta = 0.0
            current_index = getattr(self, "_current_image_index", 0)
            next_index = (current_index + 1) % len(self.images)
            self._current_image_index = next_index
            for i, sprite in enumerate(self.image_sprites):
                sprite.visible = (i == next_index)

        # Update progress bar
        self.progress_bar.width = self.width * \
            (self.completion / self.activity.duration)

        # Change style based on completion
        if is_complete:
            self.background.color = Config.SUCCESS_COLOR
            for sprite in self.image_sprites:
                sprite.opacity = 120


class StageDisplay:
    stage: Optional[Stage] = None

    def __init__(self, batch: pyglet.graphics.Batch):
        self.batch = batch
        self.columns: List[ActivityDisplay] = []
        self.column_width = (Config.window_width - len(ACTIVITY_TYPES)
                             * Config.COLUMN_GAP) / len(ACTIVITY_TYPES)
        self.column_height = Config.window_height * 0.6

    def set_data(self, stage: Optional[Stage]):
        self.stage = stage
        self.columns.clear()

        # Collect active and empty activities
        for i, activity_type in enumerate(ACTIVITY_TYPES):
            activity = next((a for a in stage.activities if a.name ==
                            activity_type), None) if stage else None
            self.columns.append(
                ActivityDisplay(
                    self.batch,
                    activity=activity,
                    x=(i + 0.5) * (self.column_width + Config.COLUMN_GAP),
                    y=Config.window_height,
                    width=self.column_width - Config.COLUMN_GAP // 2,
                    height=self.column_height,
                )
            )

    def is_complete(self) -> bool:
        return all(
            column.completion >= column.activity.duration for column in self.columns if column.activity is not None
        )

    def get_completion(self) -> float:
        if not self.stage:
            return 0.0
        return sum(column.completion for column in self.columns if column.activity) / sum(
            column.activity.duration for column in self.columns if column.activity
        )

    def update(self, dt: float, activity_name: str):
        if not self.stage:
            return

        activity = next(
            (a for a in self.columns if a.activity and a.activity.name == activity_name), None)
        if activity:
            activity.update(dt)
