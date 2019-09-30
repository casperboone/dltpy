from typing import Tuple, Type, Union, List, Optional, Any, Dict, Pattern, Match
from astor import code_gen
import ast
import re
import docstring_parser


class Function:
    """
    Representation of a parsed python function
    """

    def __init__(self, name: str, docstring: str, func_descr: Optional[str], arg_names: List[str], arg_types: List[str],
                 arg_descrs: Optional[List[str]], return_type: str, return_expr: List[str],
                 return_descr: Optional[str]) -> None:
        self.name = name
        self.docstring = docstring
        self.func_descr = func_descr
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.arg_descrs = arg_descrs
        self.return_type = return_type
        self.return_expr = return_expr
        self.return_descr = return_descr

    def __eq__(self, other) -> bool:
        if isinstance(other, Function):
            return self.name == other.name and \
                   self.docstring == other.docstring and \
                   self.arg_names == other.arg_names and \
                   self.arg_types == other.arg_types and \
                   self.return_type == other.return_type and \
                   self.return_expr == other.return_expr and \
                   self.func_descr == other.func_descr and \
                   self.arg_descrs == other.arg_descrs and \
                   self.return_descr == other.return_descr

        return False

    def __repr__(self) -> str:
        values = list(map(lambda x: repr(x), self.__dict__.values()))
        values = ",".join(values)
        return "Function(%s)" % values

    def has_types(self) -> bool:
        return any(ty for ty in self.arg_types) or self.return_type != ''

    def as_tuple(self) -> Tuple:
        return tuple(self.__dict__.values())

    def tuple_keys(self) -> Tuple:
        return tuple(self.__dict__.keys())


class Visitor(ast.NodeVisitor):

    def __init__(self) -> None:
        # fn here is tuple: (node, [return_node1, ...])
        self.fns = []
        self.return_exprs = []

    def visit_FunctionDef(self, node) -> None:
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

    def visit_Return(self, node) -> None:
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

        docstring_descr: Dict[
            str, Union[Union[Optional[str], Dict[str, str]], Optional[str]]] = self.extract_docstring_descriptions(
            self.check_docstring(docstring))
        arg_descrs: List[str] = list(map(
            lambda arg_name: docstring_descr["params"][arg_name] if arg_name in docstring_descr['params'] else '',
            arg_names
        ))

        f: Function = Function(function_name, docstring, docstring_descr["function_descr"],
                               arg_names, arg_types, arg_descrs,
                               return_type, exprs, docstring_descr["return_descr"])
        return f

    def extract_name(self, node: ast_fn_def) -> str:
        """Extract the name of the function"""
        return node.name

    def extract_docstring(self, node: ast_fn_def) -> str:
        """Extract the docstring from a function"""
        return ast.get_docstring(node) or ""

    def extract_docstring_descriptions(self, docstring: str) -> Dict[
        str, Union[Union[Optional[str], Dict[str, str]], Optional[str]]]:
        """Extract the return description from the docstring"""
        try:
            parsed_docstring: docstring_parser.parser.Docstring = docstring_parser.parse(docstring)
        except Exception:
            return {"function_descr": None, "params": {}, "return_descr": None}

        descr_map: Dict[str, Union[Union[str, Dict[str, str]], Optional[str]]] = {
            "function_descr": parsed_docstring.short_description,
            "params": {},
            "return_descr": None}

        if parsed_docstring.returns is not None:
            descr_map["return_descr"] = parsed_docstring.returns.description

        for param in parsed_docstring.params:
            descr_map["params"][param.arg_name] = param.description

        return descr_map

    def check_docstring(self, docstring: str) -> str:
        """Check the docstring if it has a valid structure for parsing and returns a valid docstring."""
        dash_line_matcher: Pattern[str] = re.compile("\s*--+")
        param_keywords: List[str] = ["Parameters", "Params", "Arguments", "Args"]
        return_keywords: List[str] = ["Returns", "Return"]
        break_keywords: List[str] = ["See Also", "Examples"]

        convert_docstring: bool = False
        add_indent: bool = False
        add_double_colon: bool = False
        active_keyword: bool = False
        end_docstring: bool = False

        preparsed_docstring: str = ""
        lines: List[str] = docstring.split("\n")
        for line in lines:
            result: Optional[Match] = re.match(dash_line_matcher, line)
            if result is not None:
                preparsed_docstring = preparsed_docstring[:-1] + ":" + "\n"
                convert_docstring = True
            else:
                for keyword in param_keywords:
                    if keyword in line:
                        add_indent = True
                        active_keyword = True
                        break
                if not active_keyword:
                    for keyword in return_keywords:
                        if keyword in line:
                            add_indent = True
                            add_double_colon = True
                            active_keyword = True
                            break
                if not add_double_colon:
                    for keyword in break_keywords:
                        if keyword in line:
                            end_docstring = True
                            break
                if end_docstring:
                    break
                if active_keyword:
                    preparsed_docstring += line + "\n"
                    active_keyword = False
                elif add_double_colon:
                    preparsed_docstring += "\t" + line + ":\n"
                    add_double_colon = False
                elif add_indent:
                    line_parts = line.split(":")
                    if len(line_parts) > 1:
                        preparsed_docstring += "\t" + line_parts[0] + "(" + line_parts[1].replace(" ", "") + "):\n"
                    else:
                        preparsed_docstring += "\t" + line + "\n"
                else:
                    preparsed_docstring += line + "\n"

        if convert_docstring:
            return preparsed_docstring
        else:
            return docstring

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
