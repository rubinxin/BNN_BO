from abc import abstractmethod

class BaseAcquisition(object):
    """
    Base class for acquisition functions. Used to define the interface
    """

    def __init__(self, model=None, verbose=False):
        self.model = model
        self.verbose = verbose

    @abstractmethod
    def _compute_acq(self, X, **kwargs):
        raise NotImplementedError
