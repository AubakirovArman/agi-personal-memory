from agim.model.wal_dual_helpers import primary_target_sequence, target_sequences


class _TokenBatch:
    def __init__(self, input_ids):
        self.input_ids = [_FakeTensor(input_ids)]


class _FakeTensor(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return _FakeTensor(value) if isinstance(item, slice) else value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        if text == "The language is":
            return _TokenBatch([10, 11, 12])
        if text == "The language is English":
            return _TokenBatch([10, 11, 12, 99])
        return _TokenBatch(self.encode(text, add_special_tokens=False))

    def encode(self, text, add_special_tokens=False):
        if text == "English":
            return [42]
        return []


class _Editor:
    tokenizer = _FakeTokenizer()


def test_both_mode_keeps_both_sequences_but_uses_contextual_primary():
    sequences = target_sequences(_Editor(), "The language is", "English", [42], "both")
    primary = primary_target_sequence(_Editor(), "The language is", "English", [42], "both")

    assert sequences == [[42], [99]]
    assert primary == [99]
