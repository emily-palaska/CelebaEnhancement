import numpy as np
# add dataset extension
class CelebADataser():
    def __init__(self, test_split=0.4, norm='min-max', verbose=False, noise=False):
        self.verbose = verbose
        self.noise = noise
        self.norm = norm

        # Place-holders
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load(self):
        if self.noise:
            # add methods for noise, portion of the dataset for basic testing
            return NotImplemented
        else:
            # actually load the dataset as y
            return NotImplemented
    
    def get_train_test_split(self):
        # return x, y train and test
        return NotImplemented
    
    def statistical_analysis(self):
        # statistical analysis similar to cifar10
        return NotImplemented
    
    def _normalize(self):  
        if self.norm == 'min-max':
            # add logic for normalization
            return NotImplemented
        else:
            # add logic for other normalization methods
            return NotImplemented

    def _create_x(self):
        # the x is degraded version of y with lower quality
        return NotImplemented             

    def _split(self):
        # here we make a split based on self.test_split with stratification
        return NotImplemented
    
    def __len__(self):
        return NotImplemented
    
    def __get_item__(self):
        return NotImplemented
