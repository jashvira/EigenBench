"""Tests for native Inspect viewer publication."""

from __future__ import annotations

import zipfile
from pathlib import Path

from numerical_rating.publish_viewer import copy_public_log


def test_copy_public_log_redacts_home_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "source.eval"
    target = tmp_path / "target.eval"
    private_path = f"{Path.home()}/code/EigenBench/run.py".encode()

    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sample.json", b'{"traceback":"' + private_path + b'"}')
        archive.writestr("attachment.bin", private_path)

    copy_public_log(source, target)

    with zipfile.ZipFile(target) as archive:
        assert archive.read("sample.json") == b'{"traceback":"<home>/code/EigenBench/run.py"}'
        assert archive.read("attachment.bin") == b"<home>/code/EigenBench/run.py"

    with zipfile.ZipFile(source) as archive:
        assert archive.read("sample.json") == b'{"traceback":"' + private_path + b'"}'
