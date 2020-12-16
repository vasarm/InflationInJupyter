
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
    pass

class ParameterDefinitionError(RuntimeError):
    pass

class WrongValueError(RuntimeError):
    pass