from typing import Tuple, Type, Union, List, Optional, Any
from astor import code_gen
import ast


class Function:
    """
    Representation of a parsed python function
    """

    def __init__(self, name, docstring, arg_names, arg_types,
                 return_type, return_expr):
        self.name = name
        self.docstring = docstring
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.return_type = return_type
        self.return_expr = return_expr

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.name == other.name and \
                   self.docstring == other.docstring and \
                   self.arg_names == other.arg_names and \
                   self.arg_types == other.arg_types and \
                   self.return_type == other.return_type and \
                   self.return_expr == other.return_expr

        return False

    def __repr__(self):
        values = list(map(lambda x: repr(x), self.__dict__.values()))
        values = ",".join(values)
        return "Function(%s)" % values

    def has_types(self):
        return any(ty for ty in self.arg_types) or self.return_type


class Visitor(ast.NodeVisitor):

    def __init__(self):
        # fn here is tuple: (node, [return_node1, ...])
        self.fns = []
        self.return_exprs = []

    def visit_FunctionDef(self, node):
        """
        When visiting (async) function definitions,
        save the main function node as well as all its return expressions
        """
        old_return_exprs = self.return_exprs
        self.return_exprs = []

        self.generic_visit(node)

        self.fns.append((node, self.return_exprs))
        self.return_exprs = old_return_exprs

    # Visiting async function is the same as sync
    # for the purpose of extracting names, types etc
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node):
        self.return_exprs.append(node)
        self.generic_visit(node)


class Extractor():
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

    def extract_fn(self, node: ast_fn_def, return_exprs: List[ast.Return]) \
            -> Function:
        function_name: str = self.extract_name(node)
        docstring: str = self.extract_docstring(node)
        (arg_names, arg_types) = self.extract_args(node)
        return_type: str = self.extract_return_type(node)
        exprs: List[str] = [self.pretty_print(re) for re in return_exprs]
        f: Function = Function(function_name, docstring,
                               arg_names, arg_types, return_type, exprs)
        return f

    def extract_name(self, node: ast_fn_def) -> str:
        """Extract the name of the function"""
        return node.name

    def extract_docstring(self, node: ast_fn_def) -> str:
        """Extract the docstring from a function"""
        return ast.get_docstring(node) or ""

    def pretty_print(self, node: Optional[ast.AST]) -> str:
        """
        Given some AST type expression,
        pretty print the type so that it is readable
        """
        if node is None:
            return ""
        return code_gen.to_source(node, indent_with="").rstrip()

    def extract_args(self, node: ast_fn_def) -> Tuple[List[str], List[str]]:
        """Extract the names and types of the function args"""
        arg_names: List[str] = [arg.arg for arg in node.args.args]
        arg_types: List[str] = [self.pretty_print(arg.annotation)
                                for arg in node.args.args]

        if node.args.vararg is not None:
            arg_names.append(node.args.vararg.arg)
            arg_types.append(self.pretty_print(node.args.vararg.annotation))

        return (arg_names, arg_types)

    def extract_return_type(self, node: ast_fn_def) -> str:
        """Extract the return type of the function"""
        return self.pretty_print(node.returns)

    def extract(self, program: str) -> List[Function]:
        """Extract useful data from python program"""
        try:
            main_node: ast.AST = ast.parse(program)
        except Exception:
            raise ParseError()

        v: Visitor = Visitor()
        v.visit(main_node)

        return list(map(lambda fn: self.extract_fn(*fn), v.fns))


class ParseError(Exception):
    pass
