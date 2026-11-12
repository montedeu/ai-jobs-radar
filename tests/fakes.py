"""Stand-ins for the model client and the Anthropic SDK, shared by the extraction tests."""

from types import SimpleNamespace

from research_radar import extraction
from research_radar.extraction import BatchRejected


class FakeModel:
    """Stands in for the model client: serves the answers it was given, and logs calls."""

    def __init__(
        self, *answers, ended=True, counted_tokens=1000, fail_submit=False, reject_submit=False
    ):
        self.answers = {a.arxiv_id: a for a in answers}
        self.batch_answers = {}  # batch id -> {arxiv_id: answer}, when batches differ
        self.fail_submit = fail_submit
        self.reject_submit = reject_submit
        self.models = []  # the model each call was given
        self.configs = []  # the request configuration each call was given
        self.sent = []  # the requests of each submitted batch
        self.ended = ended
        self.counted_tokens = counted_tokens
        self.calls = []
        self.batches = {}

    def count_tokens(self, requests, *, model, config):
        self.calls.append("count_tokens")
        self.models.append(model)
        self.configs.append(config)
        return self.counted_tokens

    def submit_batch(self, requests, *, model, config):
        self.calls.append("submit_batch")
        self.models.append(model)
        self.configs.append(config)
        if self.fail_submit:
            raise RuntimeError("connection lost")
        if self.reject_submit:
            raise BatchRejected("invalid request")
        self.sent.append(list(requests))
        batch_id = f"batch-{len(self.batches) + 1}"
        self.batches[batch_id] = [r.arxiv_id for r in requests]
        return batch_id

    def batch_ended(self, batch_id):
        self.calls.append("batch_ended")
        return self.ended

    def batch_results(self, batch_id):
        self.calls.append("batch_results")
        answers = self.batch_answers.get(batch_id, self.answers)
        return iter(answers[i] for i in self.batches[batch_id] if i in answers)


class FakeBatches:
    def __init__(self, error=None):
        self.error = error
        self.created = []

    def create(self, *, requests):
        if self.error:
            raise self.error
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_1")

    entries = ()

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        return iter(self.entries)


class FakeSdkWithBatches:
    def __init__(self, error=None):
        self.options = []
        self.batches = FakeBatches(error)
        self.messages = SimpleNamespace(batches=self.batches)

    def with_options(self, **options):
        self.options.append(options)
        return self


def real_client(sdk):
    client = extraction.AnthropicModelClient.__new__(extraction.AnthropicModelClient)
    client._client = sdk
    return client
