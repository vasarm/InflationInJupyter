class FunctionNotDefinedError(ValueError):
    """
    In case of if InflationFunction is not defined.
    """
    pass


class NoSolutionError(RuntimeError):
    """
    If module has problem getting answer.
    """
    pass


class WrongTypeError(TypeError):
    """
    Isnerted value is wrong type
    """
    pass


class ParameterDefinitionError(RuntimeError):
    """
    [summary]
    """
    pass


class WrongValueError(RuntimeError):
    """
    [summary]
    """
    pass


# Model mode is wrong (usually it is 0 and it means it cant compute)
class ModeError(RuntimeError):
    """
    [summary]
    """
    pass


class AllFunctionsNotSymbolic(RuntimeError):
    """
    Program tries to calculate symbolically but all required functions aren't symbolically defined.
    """
    pass


class FormalismError(RuntimeError):
    """
    Formalism definition is wrong. Must be "Metric" or "Palatini".
    """
    pass


class TimeoutError(TimeoutError):
    """
    If symbolic calculation takes too long time.
    """
    pass


class MethodError(RuntimeError):
    """
    Method not definide yet
    """
    pass


class ValueNotDefinedError(RuntimeError):
    """
    No value found.
    """
    pass
