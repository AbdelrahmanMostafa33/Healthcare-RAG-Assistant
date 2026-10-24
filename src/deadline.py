"""One absolute deadline shared by the API handler and the synchronous work it starts.

A cancelled asyncio task does not stop its worker thread, so the pipeline carries this deadline
through each stage instead. The provider calls also receive the time left as their HTTP timeout,
which bounds the one blocking operation that cannot check a flag itself.
"""
import time


class DeadlineExceeded(TimeoutError):
    """The request budget ran out; stop before starting more work."""


def check_deadline(deadline):
    """Raise DeadlineExceeded once an absolute time.monotonic() deadline has passed."""
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded("The request deadline passed.")


def remaining_seconds(deadline):
    """Seconds left, or None when no deadline was set."""
    if deadline is None:
        return None
    return deadline - time.monotonic()


def http_timeout_ms(deadline, ceiling_ms):
    """Per-call HTTP timeout in milliseconds, capped by the time left.

    Returns None when no deadline was set, so the provider client keeps its configured timeout.
    """
    left = remaining_seconds(deadline)
    if left is None:
        return None
    if left <= 0:
        raise DeadlineExceeded("The request deadline passed.")
    return max(1, min(ceiling_ms, int(left * 1000)))
