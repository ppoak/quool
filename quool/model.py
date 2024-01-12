import abc


class Model(abc.ABC):

    def __init__(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        self.fitted = False
    
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError
    
    def rolling_fit(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def rolling_predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def fit_predict(self, *args, **kwargs):
        raise NotImplementedError

    def rolling_fit_predict(self, *args, **kwargs):
        raise NotImplementedError
