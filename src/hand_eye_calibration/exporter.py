from __future__ import annotations

import json
from pathlib import Path

from .models import CalibrationResult, ValidationReport
from .session import CalibrationSession


class Exporter:
    def export(
        self,
        session: CalibrationSession,
        result: CalibrationResult,
        report: ValidationReport,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "calibration_type": "eye_to_hand",
            "transform_convention": "T_A_B @ p_B = p_A",
            "camera_frame": "color_optical_frame",
            "units": {"joint": "rad", "translation": "m", "depth_scale": "m_per_raw_unit"},
            "robot": session.metadata["robot"],
            "camera_calibration": session.camera_calibration.to_dict(),
            "T_base_camera": result.T_base_camera.tolist(),
            "solver": {
                "method": result.method,
                "solved_at_utc": result.solved_at_utc,
                "observation_ids": list(result.observation_ids),
            },
            "validation": report.to_dict(),
            "source_session": session.metadata["session_id"],
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return output

