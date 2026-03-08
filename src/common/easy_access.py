from master.utils.singleton import singleton
import types

@singleton
class EasyAccess:
    """ This singleton is meant to hold global variables
    (I don't want to pass parsed_args all the time)"""
    def __init__(self):
        self.model = None
        self.args = None
        self.data_shape = None
        self.loss_fn = None

    def reset(self):
        """For unit tests"""
        for attr in dir(self):
            if attr.startswith('__') is False and isinstance(self.__getattribute__(attr), types.MethodType) is False:
                self.__setattr__(attr, None)

    """
    jupyter reloads all the time and I don't restart the kernel to often
    def __setattr__(self, key, value):
        # I want these object to be constants and thus to never change
        if key in self.__dict__:
            raise Exception('Already assigned')
        super(EasyAccess, self).__setattr__(key, value)
    """