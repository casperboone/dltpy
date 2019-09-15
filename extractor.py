from typing import Tuple, Type, Union, List, Optional, Any
from astor import code_gen
import ast


class Function:
    """
    Representation of a parsed python function
    """

    def __init__(self, name, docstring, arg_names, arg_types, return_type):
        self.name = name
        self.docstring = docstring
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.return_type = return_type

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.name == other.name and \
                self.docstring == other.docstring and \
                self.arg_names == other.arg_names and \
                self.arg_types == other.arg_types and \
                self.return_type == other.return_type

        return False

    def __repr__(self):
        values = list(map(lambda x: repr(x), self.__dict__.values()))
        values = ",".join(values)
        return "Function(%s)" % values


class Extractor:
    """
    Extract data from python source code

    Example usage:
    `
        with open("example.py") as file:
            program = file.read()

        fns = Extractor().extract(program)
    `
    """

    # Only (async) function definitions contain a return value
    types_to_extract: Tuple[Type[ast.AsyncFunctionDef], Type[ast.FunctionDef]] \
        = (ast.AsyncFunctionDef, ast.FunctionDef)

    ast_fn_def: Any = Union[ast.AsyncFunctionDef, ast.FunctionDef]

    def extract_name(self, node: ast_fn_def) -> str:
        """Extract the name of the function"""
        return node.name

    def extract_docstring(self, node: ast_fn_def) -> str:
        """Extract the docstring from a function"""
        return ast.get_docstring(node) or ""

    def parse_type(self, type: Optional[ast.AST]) -> str:
        """
        Given some AST type expression,
        pretty print the type so that it is readable
        """
        if type is None:
            return ""
        return code_gen.to_source(type, indent_with="").rstrip()

    def extract_args(self, node: ast_fn_def) -> Tuple[List[str], List[str]]:
        """Extract the names and types of the function args"""
        arg_names: List[str] = [arg.arg for arg in node.args.args]
        arg_types: List[str] = [self.parse_type(arg.annotation)
                                for arg in node.args.args]

        if node.args.vararg is not None:
            arg_names.append(node.args.vararg.arg)
            arg_types.append(self.parse_type(node.args.vararg.annotation))

        return (arg_names, arg_types)

    def extract_return_type(self, node: ast_fn_def) -> str:
        """Extract the return type of the function"""
        return self.parse_type(node.returns)

    def extract(self, program: str):
        """Extract useful data from python program"""
        main_node: ast.AST = ast.parse(program)

        fns: List[Function] = []

        for node in ast.walk(main_node):
            if isinstance(node, self.types_to_extract):
                function_name: str = self.extract_name(node)
                docstring: str = self.extract_docstring(node)
                (arg_names, arg_types) = self.extract_args(node)
                return_type: str = self.extract_return_type(node)

                f: Function = Function(function_name, docstring,
                                       arg_names, arg_types, return_type)
                fns.append(f)

        return fns
