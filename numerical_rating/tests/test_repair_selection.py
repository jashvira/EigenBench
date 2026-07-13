"""Contract tests for selecting repaired response cells."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task
from inspect_ai._eval.evalset import EvalSetArgsInTaskIdentifier, task_identifier
from inspect_ai._eval.loader import resolve_task_args
from inspect_ai._eval.task.resolved import ResolvedTask
from inspect_ai.model import GenerateConfig

from numerical_rating import run_round_robin
from numerical_rating.data import (
    ResponseCell,
    load_config,
    load_repaired_cell_manifest,
    load_response_cells,
    provenance_path,
    repo_path,
    sha256_text,
)
from numerical_rating.run_pointwise import pointwise_constitution_rating


CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
MOCK_JUDGE = "mockllm/model"


def manifest_entry(cell: ResponseCell) -> dict[str, int | str]:
    """Return the repair-manifest identity for one cached response cell."""
    return {
        "scenario_index": cell.scenario_index,
        "model_id": cell.model_id,
        "model_name": cell.model_name,
        "response_hash": cell.response_hash,
    }


def inspect_task_identifier(task: Task) -> str:
    """Return Inspect's eval-set identity for a constructed task."""
    resolved = ResolvedTask(
        id="test",
        task=task,
        task_args=resolve_task_args(task),
        task_file=None,
        model=task.model,
        model_roles=None,
        sandbox=None,
        checkpoint=None,
        sequence=0,
    )
    return task_identifier(
        resolved,
        EvalSetArgsInTaskIdentifier(config=GenerateConfig()),
    )


class RepairSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)
        cls.response_cells = load_response_cells(cls.config)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_path = Path(self.temp_dir.name) / "repaired_cells.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_manifest(self, entries: list[dict[str, object]]) -> str:
        text = json.dumps({"cells": entries}, indent=2) + "\n"
        self.manifest_path.write_text(text, encoding="utf-8")
        return text

    def test_task_selects_exact_cells_and_logs_manifest_provenance(self) -> None:
        selected = [self.response_cells[9], self.response_cells[0]]
        manifest_text = self.write_manifest([manifest_entry(cell) for cell in selected])
        manifest_hash = sha256_text(manifest_text)

        task = pointwise_constitution_rating(
            config=CONFIG,
            cell_manifest=str(self.manifest_path),
            judge_model=MOCK_JUDGE,
        )

        self.assertEqual(len(task.dataset), 2)
        self.assertEqual(
            [
                (sample.metadata["scenario_index"], sample.metadata["model_id"])
                for sample in task.dataset
            ],
            [(cell.scenario_index, cell.model_id) for cell in selected],
        )
        self.assertEqual(task.version, manifest_hash)
        self.assertEqual(task.tags, ["judge:model"])
        manifest_provenance = provenance_path(self.manifest_path)
        self.assertEqual(task.metadata["cell_manifest"], manifest_provenance)
        self.assertEqual(task.metadata["cell_manifest_hash"], manifest_hash)
        for sample in task.dataset:
            self.assertEqual(sample.metadata["cell_manifest"], manifest_provenance)
            self.assertEqual(sample.metadata["cell_manifest_hash"], manifest_hash)

    def test_manifest_rejects_duplicate_mismatched_and_stale_cells(self) -> None:
        cell = self.response_cells[0]
        valid = manifest_entry(cell)
        cases = {
            "duplicate": ([valid, valid], "Duplicate repaired cell"),
            "unknown model ID": (
                [{**valid, "model_id": 999}],
                "Unexpected model ID",
            ),
            "mismatched model name": (
                [{**valid, "model_name": "Wrong model"}],
                "Model name mismatch",
            ),
            "missing cache cell": (
                [{**valid, "scenario_index": 999_999}],
                "absent from the current response cache",
            ),
            "stale response hash": (
                [{**valid, "response_hash": "0" * 16}],
                "Response hash mismatch",
            ),
        }

        for label, (entries, message) in cases.items():
            with self.subTest(label=label):
                self.write_manifest(entries)
                with self.assertRaisesRegex(ValueError, message):
                    load_repaired_cell_manifest(
                        self.manifest_path,
                        config=self.config,
                        response_cells=self.response_cells,
                    )

    def test_limit_and_manifest_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit.*cell_manifest"):
            pointwise_constitution_rating(
                config=CONFIG,
                limit=1,
                cell_manifest=str(self.manifest_path),
                judge_model=MOCK_JUDGE,
            )

    def test_manifest_hash_controls_inspect_resume_identity(self) -> None:
        entry = manifest_entry(self.response_cells[0])
        self.write_manifest([entry])
        task_kwargs = {
            "config": CONFIG,
            "cell_manifest": str(self.manifest_path),
            "judge_model": MOCK_JUDGE,
        }

        first = pointwise_constitution_rating(**task_kwargs)
        unchanged = pointwise_constitution_rating(**task_kwargs)
        self.assertEqual(
            inspect_task_identifier(first),
            inspect_task_identifier(unchanged),
        )

        changed_text = json.dumps({"source": "updated", "cells": [entry]}, indent=2)
        self.manifest_path.write_text(changed_text + "\n", encoding="utf-8")
        changed = pointwise_constitution_rating(**task_kwargs)
        self.assertNotEqual(
            inspect_task_identifier(first),
            inspect_task_identifier(changed),
        )

    def test_round_robin_normalizes_and_forwards_manifest_path(self) -> None:
        relative_manifest = "numerical_rating/repaired_cells.json"
        with patch.object(
            sys,
            "argv",
            ["run_round_robin.py", "--cell-manifest", relative_manifest],
        ):
            parsed = run_round_robin.parse_args()
        self.assertEqual(parsed.cell_manifest, repo_path(relative_manifest))

        args = argparse.Namespace(
            config=CONFIG,
            log_dir=repo_path("runs/numerical_rating/test"),
            judge=[self.config.judges[0].name],
            max_tasks=1,
            connections_per_judge=1,
            retry_attempts=1,
            retry_on_error=1,
            http_retries=1,
            timeout=10,
            generation_max_tokens=None,
            cell_manifest=repo_path(relative_manifest),
        )
        with (
            patch.object(run_round_robin, "parse_args", return_value=args),
            patch.object(run_round_robin, "load_dotenv"),
            patch.dict(os.environ, {"PETRI_OPENROUTER_API_KEY": "test-key"}),
            patch.object(
                run_round_robin,
                "pointwise_constitution_rating",
                return_value=object(),
            ) as build_task,
            patch.object(
                run_round_robin,
                "eval_set",
                return_value=(True, []),
            ) as run_eval_set,
        ):
            self.assertEqual(run_round_robin.main(), 0)

        self.assertEqual(
            build_task.call_args.kwargs["cell_manifest"],
            str(repo_path(relative_manifest)),
        )
        self.assertTrue(run_eval_set.call_args.kwargs["log_dir_allow_dirty"])


if __name__ == "__main__":
    unittest.main()
