from typing import Literal
import json


class Activity:
    name: Literal["jumpingjack", "lifting", "rowing", "running"]
    duration: float


class Stage:
    name: str
    activities: list[Activity]


class TrainingSession:
    stages: list[Stage]

    def __init__(self, path: str):
        def load_activities_from_json(activities_json):
            activities = []
            for activity_json in activities_json:
                if "name" not in activity_json or "duration" not in activity_json:
                    raise ValueError("Activity must have name and duration")
                if activity_json["name"] not in [
                    "jumpingjack",
                    "lifting",
                    "rowing",
                    "running",
                ]:
                    raise ValueError(f"Invalid activity name: {activity_json['name']}")
                activity = Activity()
                activity.name = activity_json["name"]
                activity.duration = float(activity_json["duration"])
                activities.append(activity)
            return activities

        def load_stages_from_json(stages_json):
            stages = []
            for stage_json in stages_json:
                if "name" not in stage_json or "activities" not in stage_json:
                    raise ValueError("Stage must have name and activities")
                stage = Stage()
                stage.name = stage_json["name"]
                stage.activities = load_activities_from_json(stage_json["activities"])
                stages.append(stage)
            return stages

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "stages" not in data:
                raise ValueError("Training session must have stages")

            self.stages = load_stages_from_json(data["stages"])
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {path}")
        except Exception as e:
            raise ValueError(f"Error loading training session: {e}")
