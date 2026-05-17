import argparse
import json
from pathlib import Path

from assim.assimilator import AssimilationHandler
from assim.data import datasets
from assim.methods import methods


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def main(config, config_dir):
    data_config_path = resolve_path(config["data_config"], config_dir)
    model_config_path = resolve_path(config["model_config"], config_dir)
    data_config = load_json(data_config_path)
    model_config = load_json(model_config_path)

    dataset = datasets[data_config["dataset_name"]](data_config)
    handler = AssimilationHandler(
        config=config,
        dataloader=dataset,
        method_class=methods[config["model_name"]],
        model_config=model_config,
        data_config=data_config,
    )
    handler.run_assimilation()

    out_dir = Path(config.get("outputs", {}).get("out_dir", "outputs/experiment"))
    result = {"output_dir": str(out_dir.resolve()), "metrics_path": str((out_dir / "metrics.json").resolve())}
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run a stripped 3d_var-style assimilation experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    main(config, config_path.resolve().parent)
