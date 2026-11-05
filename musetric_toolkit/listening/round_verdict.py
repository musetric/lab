import logging

LOG = logging.getLogger(__name__)


class Verdict:
    def __init__(self) -> None:
        self.failures = 0

    def fail(self, text: str) -> None:
        self.failures += 1
        LOG.error("FAIL %s", text)
