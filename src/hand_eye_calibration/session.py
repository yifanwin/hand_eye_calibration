from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from .models import CameraCalibration, CalibrationResult, DeviceInfo, Observation, ValidationReport, utc_now


class CalibrationSession:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        self.path = Path(path)
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        config_snapshot: dict[str, Any],
        robot_info: DeviceInfo,
        camera_calibration: CameraCalibration,
    ) -> "CalibrationSession":
        root = Path(path)
        if (root / "session.yaml").exists():
            raise FileExistsError(f"session already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        metadata = {
            "schema_version": cls.SCHEMA_VERSION,
            "session_id": root.name,
            "created_at_utc": utc_now(),
            "calibration_type": "eye_to_hand",
            "transform_convention": "T_A_B @ p_B = p_A",
            "camera_frame": "color_optical_frame",
            "units": {"joint": "rad", "translation": "m"},
            "robot": robot_info.to_dict(),
            "camera_calibration": camera_calibration.to_dict(),
            "config": config_snapshot,
            "runs": [],
        }
        session = cls(root, metadata)
        session._write_metadata()
        return session

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationSession":
        root = Path(path)
        with (root / "session.yaml").open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported or invalid session schema")
        CameraCalibration.from_dict(metadata["camera_calibration"])
        DeviceInfo.from_dict(metadata["robot"])
        return cls(root, metadata)

    @property
    def camera_calibration(self) -> CameraCalibration:
        return CameraCalibration.from_dict(self.metadata["camera_calibration"])

    def assert_compatible(self, robot_info: DeviceInfo, calibration: CameraCalibration) -> None:
        expected_robot = self.metadata["robot"]
        if expected_robot != robot_info.to_dict():
            raise ValueError("robot identity or frames do not match this session")
        expected_camera = self.metadata["camera_calibration"]
        actual_camera = calibration.to_dict()
        identity_keys = ("device", "color_profile", "depth_profile", "color_intrinsics", "depth_intrinsics")
        if any(expected_camera[key] != actual_camera[key] for key in identity_keys):
            raise ValueError("camera identity, active profiles, or intrinsics do not match this session")
        if expected_camera["depth_scale_m_per_unit"] != actual_camera["depth_scale_m_per_unit"]:
            raise ValueError("camera depth scale does not match this session")
        if not np.allclose(expected_camera["T_color_depth"], actual_camera["T_color_depth"], atol=1e-9):
            raise ValueError("T_color_depth does not match this session")

    def begin_run(self, *, trajectory: str, waypoint_count: int) -> str:
        run_id = uuid4().hex[:12]
        self.metadata["runs"].append({
            "run_id": run_id,
            "status": "running",
            "started_at_utc": utc_now(),
            "trajectory": trajectory,
            "attempted": 0,
            "accepted": 0,
            "waypoint_count": waypoint_count,
        })
        self._write_metadata()
        return run_id

    def finish_run(self, run_id: str, *, attempted: int, accepted: int, error: str | None = None) -> None:
        run = self._find_run(run_id)
        run.update({
            "status": "failed" if error else "completed",
            "finished_at_utc": utc_now(),
            "attempted": attempted,
            "accepted": accepted,
            "skipped": attempted - accepted,
        })
        if error:
            run["error"] = error
        self._write_metadata()

    def append_observation(self, observation: Observation) -> None:
        if observation.run_id not in {item["run_id"] for item in self.metadata["runs"]}:
            raise ValueError(f"unknown run_id: {observation.run_id}")
        target = self.path / "observations.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def observations(self) -> list[Observation]:
        target = self.path / "observations.jsonl"
        if not target.exists():
            return []
        observations: list[Observation] = []
        with target.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        observations.append(Observation.from_dict(json.loads(line)))
                    except Exception as exc:
                        raise ValueError(f"invalid observation at line {line_number}: {exc}") from exc
        return observations

    def save_result(self, result: CalibrationResult, report: ValidationReport) -> Path:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "calibration_type": "eye_to_hand",
            "transform_convention": "T_A_B @ p_B = p_A",
            "units": {"joint": "rad", "translation": "m"},
            "robot": self.metadata["robot"],
            "camera_calibration": self.metadata["camera_calibration"],
            "result": result.to_dict(),
            "validation": report.to_dict(),
        }
        target = self.path / "result.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target

    def load_result(self) -> tuple[CalibrationResult, ValidationReport]:
        data = json.loads((self.path / "result.json").read_text(encoding="utf-8"))
        result = CalibrationResult.from_dict(data["result"])
        report_data = data["validation"]
        report = ValidationReport(
            status=report_data["status"], metrics=report_data["metrics"],
            warnings=tuple(report_data.get("warnings", [])),
            outlier_observation_ids=tuple(report_data.get("outlier_observation_ids", [])),
        )
        return result, report

    def _find_run(self, run_id: str) -> dict[str, Any]:
        for run in self.metadata["runs"]:
            if run["run_id"] == run_id:
                return run
        raise ValueError(f"unknown run_id: {run_id}")

    def _write_metadata(self) -> None:
        target = self.path / "session.yaml"
        temporary = target.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.metadata, handle, sort_keys=False, allow_unicode=True)
        temporary.replace(target)
