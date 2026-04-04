"""Unit tests for ChunkedSchedule and ErrorRecovery policy wrappers."""

from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import Recover, to_wire
from positronic.policy.base import Policy, Session
from positronic.policy.harness import ChunkedSchedule, ErrorRecovery


class _ConstSession(Session):
    def __init__(self, actions):
        self._actions = actions
        self.call_count = 0

    def __call__(self, obs):
        self.call_count += 1
        return self._actions


class _ConstPolicy(Policy):
    def __init__(self, actions):
        self._actions = actions
        self._session: _ConstSession | None = None

    def new_session(self, context=None):
        self._session = _ConstSession(self._actions)
        return self._session


def _obs(now_sec=0.0, status=RobotStatus.AVAILABLE):
    return {'inference_time_ns': int(now_sec * 1e9), 'robot_state.status': status}


class TestChunkedSchedule:
    def test_first_call_runs_inference(self):
        policy = ChunkedSchedule(_ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0))
        assert result is not None
        assert len(result) == 2

    def test_returns_none_while_trajectory_active(self):
        policy = ChunkedSchedule(_ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}]))
        session = policy.new_session()
        session(_obs(now_sec=1.0))
        assert session(_obs(now_sec=1.2)) is None
        assert session(_obs(now_sec=1.4)) is None

    def test_re_infers_after_trajectory_consumed(self):
        policy = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        session = ChunkedSchedule(policy).new_session()
        session(_obs(now_sec=1.0))
        assert session(_obs(now_sec=1.3)) is None
        result = session(_obs(now_sec=1.6))
        assert result is not None
        assert policy._session.call_count == 2

    def test_no_timestamp_means_immediate_refire(self):
        policy = ChunkedSchedule(_ConstPolicy([{'v': 1}]))
        session = policy.new_session()
        session(_obs(now_sec=1.0))
        result = session(_obs(now_sec=1.01))
        assert result is not None


class TestErrorRecovery:
    def test_delegates_when_no_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(inner).new_session()
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]

    def test_emits_recover_on_first_error(self):
        session = ErrorRecovery(_ConstPolicy([{'v': 1}])).new_session()
        result = session(_obs(status=RobotStatus.ERROR))
        assert len(result) == 1
        assert result[0]['robot_command'] == to_wire(Recover())
        assert 'target_grip' not in result[0]

    def test_returns_none_on_subsequent_errors(self):
        session = ErrorRecovery(_ConstPolicy([{'v': 1}])).new_session()
        session(_obs(status=RobotStatus.ERROR))
        assert session(_obs(status=RobotStatus.ERROR)) is None
        assert session(_obs(status=RobotStatus.ERROR)) is None

    def test_resumes_after_recovery(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(inner).new_session()
        session(_obs(status=RobotStatus.ERROR))
        session(_obs(status=RobotStatus.ERROR))
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]
        assert inner._session.call_count == 1

    def test_skips_inner_during_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(inner).new_session()
        session(_obs(status=RobotStatus.AVAILABLE))
        count_before = inner._session.call_count
        session(_obs(status=RobotStatus.ERROR))
        session(_obs(status=RobotStatus.ERROR))
        assert inner._session.call_count == count_before

    def test_delegates_meta(self):
        class _MetaSession(Session):
            def __call__(self, obs):
                return []

            @property
            def meta(self):
                return {'model': 'test'}

        class _MetaPolicy(Policy):
            def new_session(self, context=None):
                return _MetaSession()

            @property
            def meta(self):
                return {'model': 'test'}

        session = ErrorRecovery(_MetaPolicy()).new_session()
        assert session.meta == {'model': 'test'}
