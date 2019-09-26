from typing import Optional, List

def add_special(self, name): ...

class Test:

    def __init__(self, x: int) -> None:
        self.x = x

    def add(self, y: int) -> int:
        """This function adds some number to y"""
        return y + self.x

    def return_optional(self, y: List[List[(int, int)]]) -> Optional[int]:
        if y[0][0] == 5:
            return None
        return y

    async def add_async(self, y: int) -> int:
        """This is an async function"""
        return await y + self.x

    def noargs() -> int:
        """This function has no input arguments"""
        return 5

    def noreturn(x: int):
        """This function has no typed return"""
        print(x)

    def return_none(x: int) -> None:
        """This function returns None"""
        print(x)

    def untyped_args(x: int, y) -> int:
        """This function has an untyped input argument"""
        return x + y

    def type_in_comments(x, y):
        # type: (int, int) -> int
        return x + y

    def with_inner(self) -> int:
        """This function has an inner function"""
        def inner() -> int:
            """This is the inner function"""
            return 12
        return inner()

    def varargs(self, msg: str, *xs: int) -> int:
        """This function has args as well as varargs"""
        sum: int = 0
        for x in xs:
            sum += x
        print(msg)
        return sum + self.x

    def untyped_varargs(self, msg: str, *xs) -> int:
        """This function has untype varargs"""
        sum: int = 0
        for x in xs:
            sum += x
        print(msg)
        return sum + self.x

    def google_docstring(self, param1: int, param2: str) -> bool:
        """Summary line.

        Extended description of function.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: Description of return value"""
        if len(param2) == param1:
            return True
        else:
            return False

    def rest_docstring(self, param1: int, param2: str) -> bool:
        """
        Summary line.

        Description of function.

        :param int param1:  The first parameter.
        :param str param2: The second parameter.
        :type param1: int
        :return: Description of return value
        :rtype: bool
        """
        if len(param2) == param1:
            return True
        else:
            return False

    def numpy_docstring(self, param1: int, param2: str) -> bool:
        """
         Summary line.

        Extended description of function.

        Parameters
        ----------
        param1 : int
            The first parameter.
        param2 : str
            The second parameter.

        Returns
        -------
        bool
            Description of return value

        See Also
        --------
        google_docstring : Same function but other docstring.

        Examples
        --------
        numpy_docstring(5, "hello")
            this will return true
        """


        if len(param2) == param1:
            return True
        else:
            return False



