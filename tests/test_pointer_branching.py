from __future__ import annotations

from copy import deepcopy

from irisu_pointer.branching import TransactionalBranches


class _Branch:
    physics_backend = "exact"

    def __init__(self, state: dict[str, int], owner: "_ExactEnv") -> None:
        self.state = deepcopy(state)
        self.owner = owner

    def state_hash(self) -> int:
        return self.state["tick"]

    def close(self) -> None:
        self.owner.closed_branches += 1


class _Checkpoint:
    def __init__(self, env: "_ExactEnv") -> None:
        self.env = env
        self.state = deepcopy(env.state)

    def branch(self) -> _Branch:
        self.env.opened_branches += 1
        return _Branch(self.state, self.env)

    def close(self) -> None:
        self.env.checkpoint_closed = True


class _ExactEnv:
    physics_backend = "exact"

    def __init__(self) -> None:
        self.state = {"tick": 7}
        self.opened_branches = 0
        self.closed_branches = 0
        self.checkpoint_closed = False
        self.clone_calls = 0

    def state_hash(self) -> int:
        return self.state["tick"]

    def fast_checkpoint(self) -> _Checkpoint:
        return _Checkpoint(self)

    def clone_state(self) -> bytes:
        self.clone_calls += 1
        return b"unused"


class _PortableEnv:
    physics_backend = "portable"

    def __init__(self) -> None:
        self.state = {"tick": 4}
        self.restores = 0

    def state_hash(self) -> int:
        return self.state["tick"]

    def clone_state(self) -> bytes:
        return str(self.state["tick"]).encode()

    def restore_state(self, snapshot: bytes) -> dict[str, int]:
        self.restores += 1
        self.state = {"tick": int(snapshot)}
        return deepcopy(self.state)


def test_exact_branches_use_fast_checkpoint_and_preserve_source() -> None:
    env = _ExactEnv()
    observation = {"tick": 7}
    with TransactionalBranches(env, observation) as branches:
        assert branches.uses_fast_checkpoint
        with branches.branch() as (branch, restored):
            branch.state["tick"] = 9
            restored["tick"] = 10
        with branches.branch() as (branch, restored):
            assert branch.state == {"tick": 7}
            assert restored == {"tick": 7}
    assert env.state == {"tick": 7}
    assert env.clone_calls == 0
    assert env.opened_branches == env.closed_branches == 2
    assert env.checkpoint_closed


def test_portable_branches_restore_before_each_branch_and_on_close() -> None:
    env = _PortableEnv()
    with TransactionalBranches(env, {"tick": 4}) as branches:
        assert not branches.uses_fast_checkpoint
        with branches.branch() as (branch, restored):
            assert branch is env
            assert restored == {"tick": 4}
            env.state["tick"] = 8
        with branches.branch() as (_branch, restored):
            assert restored == {"tick": 4}
    assert env.state == {"tick": 4}
    assert env.restores == 3
