class NotRequiredDimError(Exception):
    def __init__(self, ndim: int):
        self.ndim = ndim
    
    def __str__(self):
        return f"only {self.ndim}(s) dimension(s) allowed"


class UnfittedError(Exception):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return f"{self.name} is not fitted yet"


class RequestFailedError(Exception):
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"{self.name} request failed"