from typing import List, Optional
import pyglet
from src.config import Config
from src.training import Activity, Stage
from src.util import load_activity_images

COLUMN_PADDING = 20
ACTIVITY_TYPES = ["jumpingjack", "running", "lifting", "rowing"]


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

        # Create activity name label
        self.label = pyglet.text.Label(
            self.activity.name,
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
        label_height = self.label.content_height + margin
        available_height = self.height - label_height - (progress_label_height + margin) - (margin * 4)

        # Create and scale image sprite
        self.image_sprite = pyglet.sprite.Sprite(self.images[0], x=self.x, y=self.y - offset, batch=self.batch)
        self.target_image_height = available_height
        self.target_image_width = self.width * 0.8
        self._scale_image(available_height)

        offset += margin + self.image_sprite.height

        # Progress label
        # Format duration with appropriate time units
        if self.activity.duration < 60:
            duration_text = f"{self.activity.duration:.1f} sec"
        elif self.activity.duration < 3600:
            minutes = int(self.activity.duration // 60)
            seconds = int(self.activity.duration % 60)
            duration_text = f"{minutes}:{seconds:02d} min"
        else:
            hours = int(self.activity.duration // 3600)
            minutes = int((self.activity.duration % 3600) // 60)
            duration_text = f"{hours}:{minutes:02d} hr"

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

    def _scale_image(self, available_height: int):
        # Reset scale to original size first to get original dimensions
        self.image_sprite.scale = 1.0

        # Calculate scale factor that maintains aspect ratio and fits within available height
        scale = min(available_height / self.image_sprite.height, self.width * 0.8 / self.image_sprite.width)
        self.image_sprite.scale = scale

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
            self.image_sprite.image = self.images[next_index]
            self.image_sprite.x = self.x
            self._scale_image(self.target_image_height)

        # Update progress bar
        self.progress_bar.width = self.width * (self.completion / self.activity.duration)

        # Change style based on completion
        if is_complete:
            self.background.color = Config.SUCCESS_COLOR
            self.image_sprite.opacity = 150


class StageDisplay:
    stage: Optional[Stage] = None

    def __init__(self, batch: pyglet.graphics.Batch):
        self.batch = batch
        self.columns: List[ActivityDisplay] = []
        self.column_width = (Config.window_width - len(ACTIVITY_TYPES) * Config.COLUMN_GAP) / len(ACTIVITY_TYPES)
        self.column_height = Config.window_height * 0.7

    def set_data(self, stage: Optional[Stage]):
        self.stage = stage
        self.columns.clear()

        # Collect active and empty activities
        for i, activity_type in enumerate(ACTIVITY_TYPES):
            activity = next((a for a in stage.activities if a.name == activity_type), None) if stage else None
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

        activity = next((a for a in self.columns if a.activity and a.activity.name == activity_name), None)
        if activity:
            activity.update(dt)
