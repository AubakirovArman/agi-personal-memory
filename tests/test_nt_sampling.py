import torch

from agim.eval.easyedit_metrics import edit_nt_metrics
from agim.model.wal_dual_helpers import snapshot_rows


def test_snapshot_rows_is_deterministic_and_excludes_targets():
    weight = torch.arange(60, dtype=torch.float32).reshape(10, 6)

    first = snapshot_rows(weight, {1, 7}, sample_size=4)
    second = snapshot_rows(weight, {1, 7}, sample_size=4)

    assert list(first) == list(second)
    assert len(first) == 4
    assert not ({1, 7} & set(first))


class _FakeEditor:
    _lm_nt_snapshot = {5: torch.zeros(1), 2: torch.zeros(1)}
    _emb_nt_snapshot = {3: torch.zeros(1)}

    def measure_non_target_diffs(self):
        return {"lm_head_non_edited_max": 0.0, "embed_non_edited_max": 0.0}


def test_edit_nt_metrics_includes_sampled_row_ids():
    metrics = edit_nt_metrics(_FakeEditor(), {"lm_backup": {9: torch.zeros(1)}}, 9)

    assert metrics["lm_head_sampled_row_ids"] == [2, 5]
    assert metrics["embed_sampled_row_ids"] == [3]
    assert metrics["eos_row_changed"] is True
