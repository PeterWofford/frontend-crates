# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import unified_archives
import unified_history


def test_archive_bytes_are_deterministic_and_members_are_readable(tmp_path: Path) -> None:
    member = "families/sample/dynamo_v2-1.2.3.yaml"
    payload = b"schema_version: 2\nrecords: {}\n"
    archives = []
    for name in ("first", "second"):
        root = tmp_path / name
        family = root / "families/sample"
        family.mkdir(parents=True)
        (family / "inputs_and_golden.yaml").write_text("family: sample\ncases: {}\n")
        (root / member).write_bytes(payload)
        unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE,
                              unified_history._capture_release_sort_key)
        archives.append(next((root / "versions").iterdir()))

    assert archives[0].read_bytes() == archives[1].read_bytes()
    assert list(unified_archives.members(archives[0])) == [(member, payload)]
