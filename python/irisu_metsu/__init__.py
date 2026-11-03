"""Separate Metsu entry point; a worker path is always explicit."""

from irisu_env.exact_ipc import ExactSimulator, ExactWorkerClient


class MetsuWorkerClient(ExactWorkerClient):
    expected_backend = "exact-msvc9-r58-metsu-forward"


class MetsuSimulator(ExactSimulator):
    """Metsu exact engine with durable and fork snapshots, isolated from normal."""

    _client_type = MetsuWorkerClient

    def __init__(self, worker_path, config=None):
        if worker_path is None:
            raise ValueError("Metsu requires an explicit irisu-metsu-exact-worker path")
        super().__init__(worker_path, config=config)
