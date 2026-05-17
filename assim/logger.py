import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import init_metrics


class AssimilationLogger:
    def __init__(self, config, model_config, data_config):
        self.config = config
        self.model_config = model_config
        self.data_config = data_config
        self.out_dir = Path(config.get("outputs", {}).get("out_dir", "outputs/experiment"))
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.records = []
        self.metric_history = defaultdict(list)
        self.iteration = 0

        self.variables = data_config.get("variables")
        self.sparse_gt = bool(data_config.get("sparse_gt", False))
        self.mask = self._load_mask(data_config.get("mask_path"))
        self.metrics = init_metrics(self.mask, sparse_gt=self.sparse_gt)

        self.task = None
        self.clearml_logger = None
        clearml_config = config.get("clearml", {})
        self.use_clearml = bool(clearml_config.get("enabled", False))
        if self.use_clearml:
            try:
                from clearml import Task
            except ImportError as exc:
                raise ImportError("clearml.enabled=true, but clearml is not installed") from exc
            self.task = Task.current_task()
            if self.task is None:
                self.task = Task.init(
                    project_name=clearml_config.get("project_name") or config.get("project_name", "assimilation"),
                    task_name=clearml_config.get("task_name") or config.get("task_name", "experiment"),
                )
            self.task.connect(config, name="config")
            self.task.connect(model_config, name="model_config")
            self.task.connect(data_config, name="data_config")
            tags = clearml_config.get("tags")
            if tags:
                self.task.set_tags(list(tags))
            self.clearml_logger = self.task.get_logger()

    @staticmethod
    def _load_mask(mask_path):
        if not mask_path:
            return None
        return np.load(mask_path)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): AssimilationLogger._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AssimilationLogger._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return AssimilationLogger._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return AssimilationLogger._json_safe(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    def _field_names(self, field_count):
        if self.variables:
            return list(self.variables)
        return [f"var{i}" for i in range(field_count)]

    @staticmethod
    def _select_field(arr, index, field_count):
        arr = np.asarray(arr)
        if arr.ndim >= 3:
            return arr[index]
        if field_count == 1:
            return arr
        raise ValueError(f"Cannot select field {index} from shape {arr.shape}")

    def _compute_pair_metrics(self, pred, gt):
        return {func.__name__: func(pred, gt) for func in self.metrics}

    def log(self, meta, background, analysis, targets, obs=None):
        target_metrics = {}
        field_count = np.asarray(analysis).shape[0] if np.asarray(analysis).ndim >= 3 else 1
        field_names = self._field_names(field_count)

        for target_name, target in targets.items():
            target_metrics[target_name] = {}
            for field_index, field_name in enumerate(field_names):
                bg_field = self._select_field(background, field_index, field_count)
                an_field = self._select_field(analysis, field_index, field_count)
                gt_field = self._select_field(target, field_index, field_count)
                bg_metrics = self._compute_pair_metrics(bg_field, gt_field)
                an_metrics = self._compute_pair_metrics(an_field, gt_field)
                improvement = {
                    name: bg_metrics[name] - an_metrics[name]
                    for name in an_metrics
                    if np.isfinite(bg_metrics[name]) and np.isfinite(an_metrics[name])
                }
                target_metrics[target_name][field_name] = {
                    "background": bg_metrics,
                    "analysis": an_metrics,
                    "improvement": improvement,
                }
                self._report_case_metrics(target_name, field_name, bg_metrics, an_metrics, improvement)

        self.records.append({
            "case_id": meta.get("case_id"),
            "time": meta.get("time"),
            "metrics": target_metrics,
        })
        self.iteration += 1

    def _report_case_metrics(self, target_name, field_name, background, analysis, improvement):
        if self.clearml_logger is None:
            return
        for metric_name, value in background.items():
            if np.isfinite(value):
                self.clearml_logger.report_scalar(
                    title=f"{target_name}/{field_name}/{metric_name}",
                    series="background",
                    value=value,
                    iteration=self.iteration,
                )
        for metric_name, value in analysis.items():
            if np.isfinite(value):
                self.clearml_logger.report_scalar(
                    title=f"{target_name}/{field_name}/{metric_name}",
                    series="analysis",
                    value=value,
                    iteration=self.iteration,
                )
        for metric_name, value in improvement.items():
            if np.isfinite(value):
                self.clearml_logger.report_scalar(
                    title=f"{target_name}/{field_name}/{metric_name}",
                    series="improvement",
                    value=value,
                    iteration=self.iteration,
                )

    def _summary(self):
        grouped = defaultdict(list)
        for record in self.records:
            for target_name, target_data in record["metrics"].items():
                for field_name, field_data in target_data.items():
                    for group_name in ("background", "analysis", "improvement"):
                        for metric_name, value in field_data[group_name].items():
                            grouped[(target_name, field_name, metric_name, group_name)].append(value)

        summary = {}
        for (target_name, field_name, metric_name, group_name), values in grouped.items():
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            mean_value = float(values.mean()) if values.size else None
            summary.setdefault(target_name, {}).setdefault(field_name, {}).setdefault(metric_name, {})[
                f"{group_name}_mean"
            ] = mean_value
        return summary

    def close(self):
        payload = {
            "config": self.config,
            "model_config": self.model_config,
            "data_config": self.data_config,
            "case_metrics": self.records,
            "summary": self._summary(),
        }
        metrics_path = self.out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self._json_safe(payload), indent=2))

        if self.clearml_logger is not None:
            self._report_summary(payload["summary"])
        if self.task is not None:
            self.task.upload_artifact("metrics", str(metrics_path))
            self.task.close()

    def _report_summary(self, summary):
        for target_name, target_data in summary.items():
            for field_name, field_data in target_data.items():
                for metric_name, metric_data in field_data.items():
                    for series, value in metric_data.items():
                        if value is not None and np.isfinite(value):
                            self.clearml_logger.report_single_value(
                                name=f"{target_name}/{field_name}/{metric_name}/{series}",
                                value=value,
                            )
