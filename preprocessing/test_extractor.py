import unittest
from extractor import Extractor, Function


class TestExtractor(unittest.TestCase):


    expected = {
        "add_special" :  Function('add_special','',None,['self', 'name'],['', ''],['', ''],'',[],None),
        "__init__" :  Function('__init__','',None,['self', 'x'],['', 'int'],['', ''],'None',[],None),
        "add" :  Function('add','This function adds some number to y','This function adds some number to y',['self', 'y'],['', 'int'],['', ''],'int',['return y + self.x'],None),
        "return_optional" :  Function('return_optional','',None,['self', 'y'],['', 'List[List[int, int]]'],['', ''],'Optional[int]',['return None', 'return y'],None),
        "add_async" :  Function('add_async','This is an async function','This is an async function',['self', 'y'],['', 'int'],['', ''],'int',['return await y + self.x'],None),
        "noargs" :  Function('noargs','This function has no input arguments','This function has no input arguments',[],[],[],'int',['return 5'],None),
        "noreturn" :  Function('noreturn','This function has no typed return','This function has no typed return',['x'],['int'],[''],'',[],None),
        "return_none" :  Function('return_none','This function returns None','This function returns None',['x'],['int'],[''],'None',[],None),
        "untyped_args" :  Function('untyped_args','This function has an untyped input argument','This function has an untyped input argument',['x', 'y'],['int', ''],['', ''],'int',['return x + y'],None),
        "type_in_comments" :  Function('type_in_comments','',None,['x', 'y'],['', ''],['', ''],'',['return x + y'],None),
        "inner" :  Function('inner','This is the inner function','This is the inner function',[],[],[],'int',['return 12'],None),
        "with_inner" :  Function('with_inner','This function has an inner function','This function has an inner function',['self'],[''],[''],'int',['return inner()'],None),
        "varargs" :  Function('varargs','This function has args as well as varargs','This function has args as well as varargs',['self', 'msg', 'xs'],['', 'str', 'int'],['', '', ''],'int',['return sum + self.x'],None),
        "untyped_varargs" :  Function('untyped_varargs','This function has untype varargs','This function has untype varargs',['self', 'msg', 'xs'],['', 'str', ''],['', '', ''],'int',['return sum + self.x'],None),
        "google_docstring": Function('google_docstring','Summary line.\n\nExtended description of function.\n\nArgs:\n    param1 (int): The first parameter.\n    param2 (str): The second parameter.\n\nReturns:\n    bool: Description of return value','Summary line.',['self', 'param1', 'param2'],['', 'int', 'str'],['', 'The first parameter.', 'The second parameter.'],'bool',['return True', 'return False'],'Description of return value'),
        "rest_docstring": Function('rest_docstring','Summary line.\n\nDescription of function.\n\n:param int param1:  The first parameter.\n:param str param2: The second parameter.\n:type param1: int\n:return: Description of return value\n:rtype: bool','Summary line.',['self', 'param1', 'param2'],['', 'int', 'str'],['', 'The first parameter.', 'The second parameter.'],'bool',['return True', 'return False'],'Description of return value'),
        "numpy_docstring": Function('numpy_docstring',' Summary line.\n\nExtended description of function.\n\nParameters\n----------\nparam1 : int\n    The first parameter.\nparam2 : str\n    The second parameter.\n\nReturns\n-------\nbool\n    Description of return value\n\nSee Also\n--------\ngoogle_docstring : Same function but other docstring.\n\nExamples\n--------\nnumpy_docstring(5, "hello")\n    this will return true','Summary line.',['self', 'param1', 'param2'],['', 'int', 'str'],['', 'The first parameter.', 'The second parameter.\n        '],'bool',['return True', 'return False'],'Description of return value')
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
