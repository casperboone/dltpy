import unittest
from extractor import Extractor, Function


class TestExtractor(unittest.TestCase):

    expected = {
        "add_special" :  Function('add_special','',['self', 'name'],['', ''],'',[]),
        "__init__" :  Function('__init__','',['self', 'x'],['', 'int'],'None',[]),
        "add" :  Function('add','This function adds some number to y',['self', 'y'],['', 'int'],'int',['return y + self.x']),
        "return_optional" :  Function('return_optional','',['self', 'y'],['', 'List[List[int, int]]'],'Optional[int]',['return None', 'return y']),
        "add_async" :  Function('add_async','This is an async function',['self', 'y'],['', 'int'],'int',['return await y + self.x']),
        "noargs" :  Function('noargs','This function has no input arguments',[],[],'int',['return 5']),
        "noreturn" :  Function('noreturn','This function has no typed return',['x'],['int'],'',[]),
        "return_none" :  Function('return_none','This function returns None',['x'],['int'],'None',[]),
        "untyped_args" :  Function('untyped_args','This function has an untyped input argument',['x', 'y'],['int', ''],'int',['return x + y']),
        "type_in_comments" :  Function('type_in_comments','',['x', 'y'],['', ''],'',['return x + y']),
        "inner" :  Function('inner','This is the inner function',[],[],'int',['return 12']),
        "with_inner" :  Function('with_inner','This function has an inner function',['self'],[''],'int',['return inner()']),
        "varargs" :  Function('varargs','This function has args as well as varargs',['self', 'msg', 'xs'],['', 'str', 'int'],'int',['return sum + self.x']),
        "untyped_varargs" :  Function('untyped_varargs','This function has untype varargs',['self', 'msg', 'xs'],['', 'str', ''],'int',['return sum + self.x']),
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
