from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from assim_lib import clearml_tracking


class _FakeTaskInstance:
    id = "server-task-id"

    def get_logger(self):
        return object()

    def set_tags(self, tags):
        self.tags = tags

    def get_output_log_web_page(self):
        return "https://clearml.invalid/task/server-task-id"


def _task_type(*, offline: bool):
    instance = _FakeTaskInstance()

    class FakeTask:
        @classmethod
        def current_task(cls):
            return None

        @classmethod
        def init(cls, **kwargs):
            return instance

        @classmethod
        def is_offline(cls):
            return offline

    return FakeTask


class ClearMLOnlineEnforcementTests(unittest.TestCase):
    def _tracker(self, task_type):
        module = types.SimpleNamespace(Task=task_type)
        env = {
            "CLEARML_REQUIRE_ONLINE": "1",
            "CLEARML_API_ACCESS_KEY": "access",
            "CLEARML_API_SECRET_KEY": "secret",
            "CLEARML_API_HOST": "https://api.clearml.invalid",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(
            sys.modules, {"clearml": module}
        ), mock.patch.object(clearml_tracking, "load_clearml_env"):
            return clearml_tracking.ClearMLTracker("project", "task")

    def test_online_task_is_accepted(self) -> None:
        tracker = self._tracker(_task_type(offline=False))
        self.assertEqual(tracker.task.id, "server-task-id")

    def test_sdk_offline_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "active Task is offline"):
            self._tracker(_task_type(offline=True))

    def test_inherited_offline_environment_is_rejected_before_init(self) -> None:
        task_type = _task_type(offline=False)
        with mock.patch.dict(
            os.environ,
            {"CLEARML_REQUIRE_ONLINE": "1", "CLEARML_OFFLINE_MODE": "0"},
            clear=True,
        ), mock.patch.dict(sys.modules, {"clearml": types.SimpleNamespace(Task=task_type)}):
            with self.assertRaisesRegex(RuntimeError, "inherited"):
                clearml_tracking.ClearMLTracker("project", "task")


if __name__ == "__main__":
    unittest.main()
