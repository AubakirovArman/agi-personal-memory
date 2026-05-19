import json
import importlib.util
import types
import subprocess
from pathlib import Path


def test_build_relation_profile_map_core_logic():
    payload = {
        "summary": {
            "metrics_by_relation_id": {
                "P17": {
                    "n": 12,
                    "rephrase_all_acc": 0.20,
                    "locality_acc": 0.97,
                },
                "P20": {
                    "n": 8,
                    "rephrase_all_acc": 0.40,
                    "locality_acc": 0.88,
                },
                "P31": {
                    "n": 1,
                    "rephrase_all_acc": 0.20,
                    "locality_acc": 0.40,
                },
            }
        }
    }

    spec = importlib.util.spec_from_file_location(
        "relation_profile_map",
        Path(__file__).resolve().parents[1] / "scripts/build_relation_profile_map.py",
    )
    module = types.ModuleType("relation_profile_map")
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    build_fn = module.build_relation_profile_map
    map_data = build_fn(
        payload=payload,
        ps_threshold=0.30,
        locality_threshold=0.95,
        min_count=2,
        positive_profile="w025",
        anti_profile="target_low",
    )

    assert map_data == {
        "P17": {"positive_profile": "w025"},
        "P20": {"anti_profile": "target_low"},
    }


def test_build_relation_profile_map_cli(tmp_path: Path):
    payload_path = tmp_path / "input.json"
    output_path = tmp_path / "relation_profile_map.json"
    payload_path.write_text(
        json.dumps({
            "summary": {
                "metrics_by_relation_id": {
                    "P17": {"n": 5, "rephrase_all_acc": 0.4, "locality_acc": 0.99},
                    "P20": {"n": 5, "rephrase_all_acc": 0.1, "locality_acc": 0.9},
                },
            }
        }),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python",
            str(Path(__file__).resolve().parents[1] / "scripts/build_relation_profile_map.py"),
            "--input",
            str(payload_path),
            "--output",
            str(output_path),
            "--ps-threshold",
            "0.30",
            "--locality-threshold",
            "0.95",
            "--min-count",
            "3",
            "--positive-profile",
            "w025",
            "--anti-profile",
            "target_low",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert output_path.read_text(encoding="utf-8").strip() != ""

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == {
        "P20": {"positive_profile": "w025", "anti_profile": "target_low"},
    }
