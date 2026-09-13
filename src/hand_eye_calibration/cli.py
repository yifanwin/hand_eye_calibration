from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .collector import Collector
from .config import construct, load_config
from .exporter import Exporter
from .session import CalibrationSession
from .solver import HandEyeSolver
from .trajectory import load_trajectory, save_trajectory
from .validator import ValidationThresholds, Validator


def _thresholds(config: dict) -> ValidationThresholds:
    allowed = ValidationThresholds.__dataclass_fields__
    values = config.get("validation", {})
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown validation settings: {sorted(unknown)}")
    return ValidationThresholds(**values)


def _connect(robot, camera=None) -> None:
    robot.connect()
    try:
        if camera is not None:
            camera.start()
    except Exception:
        robot.close()
        raise


def command_teach(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    robot = construct(config["robot"])
    points = []
    robot.connect()
    try:
        print("Put the robot in freedrive mode. Press Enter to record; type q to finish.")
        while True:
            value = input(f"waypoints={len(points)}> ").strip().lower()
            if value == "q":
                break
            state = robot.read_state()
            points.append(state.joint_positions_rad.copy())
            print(f"recorded {len(points)}: {state.joint_positions_rad.tolist()}")
    finally:
        robot.close()
    if not points:
        raise ValueError("no waypoints were recorded")
    save_trajectory(args.output, np.asarray(points), robot=config["robot"]["factory"])
    print(args.output)
    return 0


def command_collect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    robot = construct(config["robot"])
    camera = construct(config["camera"])
    detector = construct(config["target"])
    _connect(robot, camera)
    try:
        calibration = camera.get_calibration()
        session_path = Path(args.session)
        if (session_path / "session.yaml").exists():
            session = CalibrationSession.load(session_path)
            session.assert_compatible(robot.get_info(), calibration)
        else:
            session = CalibrationSession.create(
                session_path,
                config_snapshot=config,
                robot_info=robot.get_info(),
                camera_calibration=calibration,
            )
        trajectory_path = args.trajectory or config["collection"].get("trajectory")
        if not trajectory_path:
            raise ValueError("trajectory path is required")
        trajectory = load_trajectory(trajectory_path, joint_count=robot.joint_count)
        collector = Collector(
            robot, camera, detector, session,
            settle_time_s=float(config["collection"].get("settle_time_s", 1.0)),
            save_images=bool(config["collection"].get("save_images", True)),
        )
        rounds = args.rounds or int(config["collection"].get("rounds", 1))
        for _ in range(rounds):
            run_id = collector.collect_run(trajectory, trajectory_name=str(trajectory_path))
            print(f"completed run {run_id}")
    finally:
        camera.close()
        robot.close()
    return 0


def command_solve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    session = CalibrationSession.load(args.session)
    observations = session.observations()
    validator = Validator(_thresholds(config))
    validator.check_solvable(observations)
    result = HandEyeSolver(config["solver"].get("method", "park")).solve(observations)
    report = validator.validate(observations, result)
    path = session.save_result(result, report)
    print(f"{report.status}: {path}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    session = CalibrationSession.load(args.session)
    result, _ = session.load_result()
    report = Validator(_thresholds(config)).validate(session.observations(), result)
    session.save_result(result, report)
    print(report.status)
    for name, value in report.metrics.items():
        print(f"{name}: {value}")
    return 0


def command_export(args: argparse.Namespace) -> int:
    session = CalibrationSession.load(args.session)
    result, report = session.load_result()
    path = Exporter().export(session, result, report, args.output)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handeye")
    commands = parser.add_subparsers(dest="command", required=True)
    teach = commands.add_parser("teach", help="record a radian joint trajectory")
    teach.add_argument("--config", required=True)
    teach.add_argument("--output", required=True)
    teach.set_defaults(handler=command_teach)
    collect = commands.add_parser("collect", help="append observations to a session")
    collect.add_argument("--config", required=True)
    collect.add_argument("--session", required=True)
    collect.add_argument("--trajectory")
    collect.add_argument("--rounds", type=int)
    collect.set_defaults(handler=command_collect)
    solve = commands.add_parser("solve", help="jointly solve all session observations")
    solve.add_argument("--config", required=True)
    solve.add_argument("--session", required=True)
    solve.set_defaults(handler=command_solve)
    validate = commands.add_parser("validate", help="validate an existing result")
    validate.add_argument("--config", required=True)
    validate.add_argument("--session", required=True)
    validate.set_defaults(handler=command_validate)
    export = commands.add_parser("export", help="write canonical calibration JSON")
    export.add_argument("--session", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(handler=command_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
