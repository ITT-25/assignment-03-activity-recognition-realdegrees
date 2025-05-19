from typing import List, Optional
import pyglet
from src.config import Config
from typing import TYPE_CHECKING

from src.util import lerp_colors, load_activity_images

if TYPE_CHECKING:
    from src.training import TrainingSession, Stage
    from src.table import StageDisplay


class ActivityMeter:
    start_color = Config.ERROR_COLOR
    middle_color = Config.WARNING_COLOR
    end_color = Config.SUCCESS_COLOR

    def __init__(self, batch: pyglet.graphics.Batch, width: int = 200, height: int = 20, x: int = 0, y: int = 0):
        self.width = width
        self.height = height
        self.label = pyglet.text.Label(
            "Activity Meter",
            font_size=14,
            x=x + width // 2,
            y=y,
            anchor_x="center",
            anchor_y="top",
            batch=batch,
            color=Config.TEXT_COLOR,
        )
        y -= self.label.font_size * 2 + 2
        self.meter_background = pyglet.shapes.Rectangle(
            x=x,
            y=y,
            width=self.width,
            height=self.height,
            color=Config.SECONDARY_COLOR,
            batch=batch,
        )
        self.meter_foreground = pyglet.shapes.Rectangle(
            x=x,
            y=y,
            width=0,
            height=self.height,
            color=self.start_color,
            batch=batch,
        )
        self.threshold_indicator = pyglet.shapes.Rectangle(
            x=x + self.width * 0.95,
            y=y + (self.height * 0.3) / 2,
            width=3,
            height=self.height * 1.3,
            color=Config.ERROR_COLOR,
            batch=batch,
        )
        self.meter_foreground.anchor_x = 0
        self.meter_foreground.anchor_y = self.meter_foreground.height
        self.meter_background.anchor_x = 0
        self.meter_background.anchor_y = self.meter_background.height
        self.threshold_indicator.anchor_y = self.threshold_indicator.height
        y -= self.meter_background.height + 4
        self.info_text = pyglet.text.Label(
            "Complete the exercises on the screen to advance the training session.",
            font_size=8,
            x=x + width // 2,
            y=y,
            anchor_x="center",
            anchor_y="top",
            batch=batch,
            color=Config.TEXT_COLOR,
        )

    def update(self, dt: float, activity_rate: float):
        target = self.width * activity_rate
        smoothing = 5.0

        # Linear interpolation towards target width
        self.meter_foreground.width += (target - self.meter_foreground.width) * min(dt * smoothing, 1.0)
        self.meter_foreground.color = lerp_colors(
            [self.start_color, self.middle_color, self.middle_color, self.end_color, self.end_color, self.end_color],
            self.meter_foreground.width / self.width,
        )


class StagePreviewDisplay:
    width = 100
    height = 20
    progress = 0.0

    def __init__(self, batch: pyglet.graphics.Batch, stage: Optional["Stage"] = None, x: int = 0, y: int = 0):
        self.stage = stage

        self.background = pyglet.shapes.Rectangle(
            x=x,
            y=y,
            width=self.width,
            height=self.height,
            color=Config.SECONDARY_COLOR,
            batch=batch,
        )
        self.progress_bar = pyglet.shapes.Rectangle(
            x=x,
            y=y,
            width=0,
            height=self.height,
            color=Config.SUCCESS_COLOR,
            batch=batch,
        )
        self.background.anchor_x = 0
        self.progress_bar.anchor_x = 0
        self.background.anchor_y = self.height
        self.progress_bar.anchor_y = self.height
        self.label = pyglet.text.Label(
            stage.name,
            font_size=10,
            x=x + 2,
            y=y - 2,
            anchor_x="left",
            anchor_y="top",
            batch=batch,
            color=Config.TEXT_COLOR,
        )
        self.activity_preview_images: List[pyglet.image.AbstractImage] = [
            load_activity_images(activity.name)[0] for activity in stage.activities
        ]
        for image in self.activity_preview_images:
            image.anchor_x = image.width
            image.anchor_y = image.height

        self.images = []
        x_offset = x + self.width + 10

        for i, image in enumerate(self.activity_preview_images):
            # Create sprite with original image dimensions
            sprite = pyglet.sprite.Sprite(image, x=x + x_offset, y=y, batch=batch)

            scale_factor = (self.height * 0.9) / image.height
            sprite.scale = scale_factor

            self.images.append(sprite)

            # Update x_offset for next sprite
            x_offset += sprite.width + 2

    def set_progress(self, progress: float):
        self.progress = progress
        self.progress_bar.width = self.width * progress


class SessionInfoDisplay:
    def __init__(
        self, batch: pyglet.graphics.Batch, session: Optional["TrainingSession"], stage_display: "StageDisplay"
    ):
        self.batch = batch
        self.session = session
        self.stage_display = stage_display
        self.info_area_height = Config.window_height * 0.4  # Bottom 30% of screen

        margin = 10
        gap = 2

        self.stage_complete_label = pyglet.text.Label(
            "",
            font_size=20,
            x=Config.window_width - margin,
            y=margin,
            anchor_x="right",
            anchor_y="bottom",
            batch=batch,
            color=Config.SUCCESS_COLOR,
        )
        self.stage_complete_label.visible = False

        # Initialize the stage overview displays

        self.stage_overview_label = pyglet.text.Label(
            "Training Session Overview",
            font_size=12,
            x=margin,
            y=self.info_area_height - margin,
            anchor_x="left",
            anchor_y="top",
            batch=batch,
            color=Config.TEXT_COLOR,
        )
        self.stage_overview_displays = [
            StagePreviewDisplay(
                self.batch,
                stage,
                x=margin,
                y=self.info_area_height
                - margin
                - self.stage_overview_label.font_size * 2
                - i * (StagePreviewDisplay.height + gap),
            )
            for i, stage in enumerate(session.stages)
        ]

        # Initialize Activity Meter
        meter_width = 400
        self.activity_meter = ActivityMeter(
            self.batch, meter_width, 30, Config.window_width - margin - meter_width, self.info_area_height - margin
        )

    def update(self, dt: float):
        self.stage_complete_label.visible = self.stage_display.is_complete()
        self.stage_complete_label.text = (
            f"Stage {self.stage_display.stage.name} complete!"
            if not all(sod.progress >= 1.0 for sod in self.stage_overview_displays)
            else "Training complete!"
        )
        for stage_overview_display in self.stage_overview_displays:
            if stage_overview_display.stage is None or self.stage_display.stage is None:
                continue

            if self.stage_display.stage.name == stage_overview_display.stage.name:
                stage_overview_display.set_progress(self.stage_display.get_completion())
