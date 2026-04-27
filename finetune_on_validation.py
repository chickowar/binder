#!/usr/bin/env python3
"""
Fine-tune a Binder model on validation data instead of the original train split.

The script reuses the existing BinderTraining pipeline by generating a temporary
config file with:
- train_file replaced by validation data
- evaluation/prediction disabled
- optional checkpoint/epoch/output overrides

Validation data can come either from:
1. `validation_file` in the source config
2. an internal split reproduced from `train_file` using
   `validation_split_ratio` + `validation_split_seed`
"""

import argparse
import json
import os
import tempfile
from typing import Any, Dict

from datasets import load_dataset

from train_binder import BinderTraining


PATH_KEYS = [
    "train_file",
    "validation_file",
    "test_file",
    "entity_type_file",
    "output_dir",
    "cache_dir",
    "logging_dir",
    "binder_model_name_or_path",
    "config_name",
    "tokenizer_name",
    "model_name_or_path",
    "resume_from_checkpoint",
]


def _resolve_config_paths(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    config = dict(config)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    for key in PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or value == "":
            continue
        if os.path.isabs(value):
            continue
        if "/" in value and not any(sep in value for sep in ("./", ".\\", "../", "..\\")):
            continue
        config[key] = os.path.abspath(os.path.join(config_dir, value))

    return config


def _dataset_loader_extension(path: str) -> str:
    extension = path.rsplit(".", 1)[-1].lower()
    if extension in {"json", "jsonl"}:
        return "json"
    return extension


def _dump_jsonl_dataset(dataset, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_validation_train_file(config: Dict[str, Any], source_config_path: str) -> str:
    validation_file = config.get("validation_file")
    if validation_file:
        return validation_file

    train_file = config.get("train_file")
    split_ratio = config.get("validation_split_ratio")
    split_seed = config.get("validation_split_seed", 42)

    if not train_file or split_ratio is None:
        raise ValueError(
            "Config must define either `validation_file` or "
            "`train_file` + `validation_split_ratio`."
        )

    dataset = load_dataset(
        _dataset_loader_extension(train_file),
        data_files={"train": train_file},
        cache_dir=config.get("cache_dir"),
    )["train"]
    split_dataset = dataset.train_test_split(test_size=split_ratio, seed=split_seed)
    validation_dataset = split_dataset["test"]

    config_dir = os.path.dirname(os.path.abspath(source_config_path))
    fd, temp_path = tempfile.mkstemp(
        prefix="validation_split_",
        suffix=".jsonl",
        dir=config_dir,
    )
    os.close(fd)
    _dump_jsonl_dataset(validation_dataset, temp_path)
    return temp_path


def _prepare_finetune_config(
    base_config: Dict[str, Any],
    source_config_path: str,
    epochs: float | None,
    output_dir: str | None,
    binder_checkpoint: str | None,
    resume_from_checkpoint: str | None,
    learning_rate: float | None,
    run_name_suffix: str | None,
) -> Dict[str, Any]:
    config = _resolve_config_paths(base_config, source_config_path)
    validation_train_file = _build_validation_train_file(config, source_config_path)

    config["train_file"] = validation_train_file
    config["validation_file"] = None
    config["validation_split_ratio"] = None
    config["test_file"] = None
    config["do_eval"] = False
    config["do_predict"] = False
    config["load_best_model_at_end"] = False
    config["metric_for_best_model"] = None
    config["greater_is_better"] = None
    config["evaluation_strategy"] = "no"
    config["eval_strategy"] = "no"
    config["eval_steps"] = None

    if epochs is not None:
        config["num_train_epochs"] = epochs
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if binder_checkpoint is not None:
        config["binder_model_name_or_path"] = os.path.abspath(binder_checkpoint)
    if resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = os.path.abspath(resume_from_checkpoint)

    base_output_dir = config.get("output_dir")
    if output_dir is not None:
        config["output_dir"] = os.path.abspath(output_dir)
    elif base_output_dir:
        config["output_dir"] = os.path.abspath(base_output_dir.rstrip("\\/") + "_val_finetune")

    if run_name_suffix:
        base_run_name = config.get("run_name", "binder-validation-finetune")
        config["run_name"] = f"{base_run_name}-{run_name_suffix}"

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Binder on validation data using an existing training config."
    )
    parser.add_argument("config", help="Path to the source training JSON config.")
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override num_train_epochs for the validation fine-tuning run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir for the validation fine-tuning run.",
    )
    parser.add_argument(
        "--binder-checkpoint",
        default=None,
        help="Load Binder weights from this checkpoint before fine-tuning.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume optimizer/scheduler/trainer state from a Trainer checkpoint.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning_rate for the validation fine-tuning run.",
    )
    parser.add_argument(
        "--run-name-suffix",
        default="val-ft",
        help="Suffix appended to run_name.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = json.load(handle)

    finetune_config = _prepare_finetune_config(
        base_config=base_config,
        source_config_path=config_path,
        epochs=args.epochs,
        output_dir=args.output_dir,
        binder_checkpoint=args.binder_checkpoint,
        resume_from_checkpoint=args.resume_from_checkpoint,
        learning_rate=args.learning_rate,
        run_name_suffix=args.run_name_suffix,
    )

    config_dir = os.path.dirname(config_path)
    fd, temp_config_path = tempfile.mkstemp(
        prefix="validation_finetune_",
        suffix=".json",
        dir=config_dir,
    )
    os.close(fd)
    with open(temp_config_path, "w", encoding="utf-8") as handle:
        json.dump(finetune_config, handle, ensure_ascii=False, indent=2)

    print(f"Temporary validation fine-tune config: {temp_config_path}")
    print(f"Training on validation data from: {finetune_config['train_file']}")
    print(f"Output dir: {finetune_config.get('output_dir')}")
    if finetune_config.get("binder_model_name_or_path"):
        print(f"Binder checkpoint: {finetune_config['binder_model_name_or_path']}")
    if finetune_config.get("resume_from_checkpoint"):
        print(f"Resume checkpoint: {finetune_config['resume_from_checkpoint']}")

    trainer = BinderTraining(config_path=temp_config_path)
    trainer.train_only()


if __name__ == "__main__":
    main()
