from unittest import TestCase

from helpers.file_helper import get_ott_negative

class TestFileHelper(TestCase):
    def test_ott_negative(self):
        ott_negative = get_ott_negative()
        number_of_negative = len(get_ott_negative())
        self.assertEqual(number_of_negative, 800)