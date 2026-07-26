from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REQUIRED_ENV = (
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
    "CLEARML_API_HOST",
)


def _debug(message: str) -> None:
    print(f"[assim_lib][clearml] {message}", flush=True)


def load_clearml_env(env_path: str | None = None) -> None:
    if env_path:
        loaded = load_dotenv(env_path, override=False)
        _debug(f"load_dotenv path={env_path} loaded={loaded}")
        return

    # Works both from the repository root and from the Docker WORKDIR /home.
    loaded = load_dotenv(override=False)
    _debug(f"load_dotenv default loaded={loaded}")


def validate_clearml_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    present = [name for name in REQUIRED_ENV if os.environ.get(name)]
    _debug(f"required env present={present} missing={missing}")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing ClearML environment variables: {joined}. "
            "Create .env from .env.example or export these variables before running."
        )


class ClearMLTracker:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: list[str] | tuple[str, ...] | None = None,
        output_uri: str | None = None,
        env_path: str | None = None,
    ):
        _debug(f"initializing project={project_name} task={task_name}")
        load_clearml_env(env_path)
        validate_clearml_env()
        try:
            from clearml import Task
        except ImportError as exc:
            raise ImportError(
                "ClearML support requires the clearml package. Install requirements.txt."
            ) from exc

        task = Task.current_task()
        if task is None:
            task = Task.init(project_name=project_name, task_name=task_name, output_uri=output_uri)
            _debug("created new ClearML task")
        else:
            _debug("using current ClearML task")
        self.task = task
        _debug(f"task id={getattr(self.task, 'id', None)}")
        try:
            _debug(f"task url={self.task.get_output_log_web_page()}")
        except Exception:
            pass
        if tags:
            self.task.set_tags(list(tags))
            _debug(f"tags={list(tags)}")
        self.logger = self.task.get_logger()

    def connect(self, name: str, payload: dict[str, Any]) -> None:
        _debug(f"connect config name={name}")
        self.task.connect(payload, name=name)

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.logger.report_scalar(title=title, series=series, value=float(value), iteration=int(iteration))

    def report_single_value(self, name: str, value: float) -> None:
        self.logger.report_single_value(name=name, value=float(value))

    def report_figure(self, title: str, series: str, figure, iteration: int) -> None:
        try:
            self.logger.report_matplotlib_figure(
                title=title,
                series=series,
                figure=figure,
                iteration=int(iteration),
                report_interactive=False,
            )
        except TypeError:
            self.logger.report_matplotlib_figure(
                title=title,
                series=series,
                figure=figure,
                iteration=int(iteration),
            )

    def report_image(self, title: str, series: str, path: str | Path, iteration: int) -> None:
        path = Path(path)
        if path.exists():
            _debug(f"report image title={title} series={series} path={path}")
            self.logger.report_image(
                title=title,
                series=series,
                local_path=str(path),
                iteration=int(iteration),
            )
        else:
            _debug(f"skip missing image title={title} series={series} path={path}")

    def upload_artifact(self, name: str, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            _debug(f"upload artifact name={name} path={path}")
            self.task.upload_artifact(name=name, artifact_object=str(path))
        else:
            _debug(f"skip missing artifact name={name} path={path}")

    def close(self) -> None:
        _debug("closing task")
        self.task.close()
