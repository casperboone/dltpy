import unittest
from extractor import Extractor, Function


class TestExtractor(unittest.TestCase):


    expected = {
        "add_special" :  Function('add_special','',['self', 'name'],['', ''],'',[], None, {}, None),
        "__init__" :  Function('__init__','',['self', 'x'],['', 'int'],'None',[], None, {}, None),
        "add" :  Function('add','This function adds some number to y',['self', 'y'],['', 'int'],'int',['return y + self.x'], "This function adds some number to y", {}, None),
        "return_optional" :  Function('return_optional','',['self', 'y'],['', 'List[List[int, int]]'],'Optional[int]',['return None', 'return y'], None, {}, None),
        "add_async" :  Function('add_async','This is an async function',['self', 'y'],['', 'int'],'int',['return await y + self.x'], "This is an async function", {}, None),
        "noargs" :  Function('noargs','This function has no input arguments',[],[],'int',['return 5'], "This function has no input arguments", {}, None),
        "noreturn" :  Function('noreturn','This function has no typed return',['x'],['int'],'',[], "This function has no typed return", {}, None),
        "return_none" :  Function('return_none','This function returns None',['x'],['int'],'None',[], "This function returns None", {}, None),
        "untyped_args" :  Function('untyped_args','This function has an untyped input argument',['x', 'y'],['int', ''],'int',['return x + y'], "This function has an untyped input argument", {}, None),
        "type_in_comments" :  Function('type_in_comments','',['x', 'y'],['', ''],'',['return x + y'], None, {}, None),
        "inner" :  Function('inner','This is the inner function',[],[],'int',['return 12'], "This is the inner function", {}, None),
        "with_inner" :  Function('with_inner','This function has an inner function',['self'],[''],'int',['return inner()'], "This function has an inner function", {}, None),
        "varargs" :  Function('varargs','This function has args as well as varargs',['self', 'msg', 'xs'],['', 'str', 'int'],'int',['return sum + self.x'], "This function has args as well as varargs", {}, None),
        "untyped_varargs" :  Function('untyped_varargs','This function has untype varargs',['self', 'msg', 'xs'],['', 'str', ''],'int',['return sum + self.x'], "This function has untype varargs", {}, None),
        "google_docstring": Function('google_docstring','Summary line.\n\nExtended description of function.\n\nArgs:\n    param1 (int): The first parameter.\n    param2 (str): The second parameter.\n\nReturns:\n    bool: Description of return value',['self', 'param1', 'param2'],['', 'int', 'str'],'bool',['return True', 'return False'],'Summary line.',{'param1': 'The first parameter.', 'param2': 'The second parameter.'},'Description of return value'),
        "rest_docstring": Function('rest_docstring','Summary line.\n\nDescription of function.\n\n:param int param1:  The first parameter.\n:param str param2: The second parameter.\n:type param1: int\n:return: Description of return value\n:rtype: bool',['self', 'param1', 'param2'],['', 'int', 'str'],'bool',['return True', 'return False'],'Summary line.',{'param1': 'The first parameter.', 'param2': 'The second parameter.'},'Description of return value'),
        "numpy_docstring": Function('numpy_docstring',' Summary line.\n\nExtended description of function.\n\nParameters\n----------\nparam1 : int\n    The first parameter.\nparam2 : str\n    The second parameter.\n\nReturns\n-------\nbool\n    Description of return value\n\nSee Also\n--------\ngoogle_docstring : Same function but other docstring.\n\nExamples\n--------\nnumpy_docstring(5, "hello")\n    this will return true',['self', 'param1', 'param2'],['', 'int', 'str'],'bool',['return True', 'return False'],'Summary line.',{'param1': 'The first parameter.', 'param2': 'The second parameter.\n        '},'Description of return value')
    }

    def setUp(self):
        with open("./resources/example.py") as file:
            program = file.read()
        self.fns = Extractor().extract(program)

    def test_function_parsing(self):
        for fn in self.expected.keys():
            actual = [x for x in self.fns if x.name == fn][0]
            expected = self.expected[fn]
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
