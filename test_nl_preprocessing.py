import unittest
from extractor import Function
from nl_preprocessing import NLPreprocessor, CleanFunction


class TestExtractor(unittest.TestxCase):
    def test_preprocessing_a_function(self):
        function = Function(
            'add_two_numbers',
            'This function adds some number to y',
            ['self', 'first_number', 'second_number'],
            ['', 'SpecialNumberType', 'int'],
            'int',
            ['return first_number + second_number']
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(repr(CleanFunction(
            function,
            'add two number',
            'function add number',
            ['self', 'first number', 'second number'],
            ['', 'SpecialNumberType', 'int'],
            'int',
            ['first number second number']
        )), repr(result))

    def test_preprocessing_a_function_with_camel_case_name(self):
        function = Function(
            'addTwoNumbers',
            'This function adds some number to y',
            ['self', 'first_number', 'second_number'],
            ['', 'SpecialNumberType', 'int'],
            'int',
            ['return first_number + second_number']
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(repr(CleanFunction(
            function,
            'add two number',
            'function add number',
            ['self', 'first number', 'second number'],
            ['', 'SpecialNumberType', 'int'],
            'int',
            ['first number second number']
        )), repr(result))

    def test_preprocessing_a_function_without_arguments_and_return_statement(self):
        function = Function(
            'add_two_numbers',
            'This function adds some number to y',
            [],
            [],
            'None',
            []
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(repr(CleanFunction(
            function,
            'add two number',
            'function add number',
            [],
            [],
            'None',
            []
        )), repr(result))

    def test_preprocessing_a_function_with_a_lot_of_punctuation_and_text(self):
        function = Function(
            'validate_clip_with_axis',
            """
            If 'NDFrame.clip' is called via the numpy library, the third
            parameter in its signature is 'out'. Which can takes an ndarray,
            so check if the 'axis' parameter is an instance of ndarray? Since
            'axis' itself should either be an integer or None
            """,
            ['axis', 'args', 'kwargs'],
            ['ndarray', 'list', 'list'],
            'ndarray',
            ['axis']
        )

        result = NLPreprocessor().preprocess(function)

        self.assertEqual(repr(CleanFunction(
            function,
            'validate clip with axis',
            'nd frame clip call via numpy library third parameter signature . take ndarray check axis parameter '
            + 'instance ndarray. since axis either integer none',
            ['axis', 'args', 'kwargs'],
            ['ndarray', 'list', 'list'],
            'ndarray',
            ['axis']
        )), repr(result))


if __name__ == '__main__':
    unittest.main()
