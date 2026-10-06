"""Backend-neutral transactional branches for pointer teachers.

Exact environments use the worker's fork/COW checkpoint when it is available,
so each branch has isolated process ownership and the live source is untouched.
Portable environments, and exact test doubles without the fast API, retain the
clone/restore protocol.  The latter is correct for exact workers but replays the
saved action history and is therefore the slower compatibility path.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class TransactionalBranches:
    """Own one source checkpoint and yield independent sequential branches."""

    def __init__(
        self,
        env: Any,
        observation: Mapping[str, Any],
        *,
        snapshot: bytes | None = None,
    ) -> None:
        backend = getattr(env, "physics_backend", None)
        if backend not in {"portable", "exact"}:
            raise ValueError("branch environment must use portable or exact physics")
        self._env = env
        self._observation = copy.deepcopy(observation)
        self._backend = backend
        state_hash = getattr(env, "state_hash", None)
        self._state_hash = state_hash if callable(state_hash) else None
        self._source_hash = self._state_hash() if self._state_hash else None
        self._checkpoint: Any | None = None
        self._snapshot = snapshot

    @property
    def uses_fast_checkpoint(self) -> bool:
        return self._checkpoint is not None

    def __enter__(self) -> TransactionalBranches:
        fast_checkpoint = getattr(self._env, "fast_checkpoint", None)
        if self._backend == "exact" and callable(fast_checkpoint):
            self._checkpoint = fast_checkpoint()
            return self
        restore = getattr(self._env, "restore_state", None)
        if not callable(restore):
            raise TypeError("branch environment lacks restore capability")
        if self._snapshot is None:
            clone = getattr(self._env, "clone_state", None)
            if not callable(clone):
                raise TypeError("branch environment lacks clone capability")
            self._snapshot = clone()
        return self

    def _verify_hash(self, env: Any) -> None:
        if self._source_hash is None:
            return
        state_hash = getattr(env, "state_hash", None)
        if not callable(state_hash) or state_hash() != self._source_hash:
            raise RuntimeError("branch did not begin at the source state")

    @contextmanager
    def branch(self) -> Iterator[tuple[Any, Mapping[str, Any]]]:
        if self._checkpoint is not None:
            branch = self._checkpoint.branch()
            try:
                self._verify_hash(branch)
                yield branch, copy.deepcopy(self._observation)
            finally:
                branch.close()
            return
        if self._snapshot is None:
            raise RuntimeError("transactional branches are not open")
        restored = self._env.restore_state(self._snapshot)
        if not isinstance(restored, Mapping):
            raise TypeError("restore_state must return a public mapping")
        self._verify_hash(self._env)
        yield self._env, restored

    def close(self) -> None:
        if self._checkpoint is not None:
            self._checkpoint.close()
            self._checkpoint = None
            self._verify_hash(self._env)
            return
        if self._snapshot is not None:
            self._env.restore_state(self._snapshot)
            self._verify_hash(self._env)
            self._snapshot = None

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["TransactionalBranches"]
