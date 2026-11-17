import unittest

from scripts.restore_day6_snapshot import validate_target


class RestoreGuardTests(unittest.TestCase):
    def test_refuses_original_day6_and_nonlocal_or_wrong_database(self):
        for target in ('postgresql://u:p@127.0.0.1:55433/day6_lab',
                       'postgresql://u:p@127.0.0.1:55434/day6_lab',
                       'postgresql://u:p@example.com:55434/day7_lab',''):
            with self.assertRaises(ValueError):
                validate_target(target)

    def test_accepts_only_the_separate_local_restore_destination(self):
        validate_target('postgresql://u:p@127.0.0.1:55434/day7_lab')
