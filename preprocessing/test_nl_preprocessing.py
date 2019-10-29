import unittest
from preprocessing.extractor import Function
from preprocessing.nl_preprocessing import NLPreprocessor


class TestExtractor(unittest.TestCase):
    def test_preprocessing_a_function(self):
        function = Function(
            'add_two_numbers',
            'This function adds some number to y',
            None,
            ['self', 'first_number', 'second_number'],
            ['', 'SpecialNumberType', 'int'],
            ['', '', ''],
            'int',
            ['return first_number + second_number'],
            None
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(Function(
            'add two number',
            'function add number',
            None,
            ['self', 'first number', 'second number'],
            ['', 'SpecialNumberType', 'int'],
            ['', '', ''],
            'int',
            ['first number second number'],
            None
        ), result)

    def test_preprocessing_a_function_with_camel_case_name(self):
        function = Function(
            'addTwoNumbers',
            'This function adds some number to y',
            None,
            ['self', 'first_number', 'second_number'],
            ['', 'SpecialNumberType', 'int'],
            ['', '', ''],
            'int',
            ['return first_number + second_number'],
            None
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(Function(
            'add two number',
            'function add number',
            None,
            ['self', 'first number', 'second number'],
            ['', 'SpecialNumberType', 'int'],
            ['', '', ''],
            'int',
            ['first number second number'],
            None
        ), result)

    def test_preprocessing_a_function_without_arguments_and_return_statement(self):
        function = Function(
            'add_two_numbers',
            'This function adds some number to y',
            None,
            [],
            [],
            ['', '', ''],
            'None',
            [],
            None
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(Function(
            'add two number',
            'function add number',
            None,
            [],
            [],
            ['', '', ''],
            'None',
            [],
            None
        ), result)

    def test_preprocessing_a_function_with_a_lot_of_punctuation_and_text(self):
        function = Function(
            'validate_clip_with_axis',
            """
            Summary line.
        
            If 'NDFrame.clip' is called via the numpy library, the third
            parameter in its signature is 'out'. Which can takes an ndarray,
            so check if the 'axis' parameter is an instance of ndarray? Since
            'axis' itself should either be an integer or None

            :param ndarray axis:  The first parameter.
            :param list args: The second parameter.
            :return: Description of return value
            :rtype: bool
            """,
            'Summary line.',
            ['axis', 'args', 'kwargs'],
            ['ndarray', 'list', 'list'],
            ['The first parameter', 'The second parameter', ''],
            'ndarray',
            ['axis'],
            'Description of return value'
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(Function(
            'validate clip with axis',
            'summary line nd frame clip call via numpy library third parameter signature take ndarray check axis '
            + 'parameter instance ndarray since axis either integer none param ndarray axis first parameter param '
            + 'list args second parameter return description return value rtype bool',
            'summary line',
            ['axis', 'args', 'kwargs'],
            ['ndarray', 'list', 'list'],
            ['first parameter', 'second parameter', ''],
            'ndarray',
            ['axis'],
            'description return value'
        ), result)


if __name__ == '__main__':
    unittest.main()
