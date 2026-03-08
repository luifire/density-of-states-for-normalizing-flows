from survae.model.data.datasets.image import UnsupervisedMNIST
from torchvision.transforms import Compose
from survae.model.data import TrainTestLoader


class MNIST_MASTER(TrainTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    '''

    def __init__(self, root=None, download=True, num_bits=8, pil_transforms=[], eval_transformation=[]):

        self.root = root

        #print(trans_train)
        #print('#########')
        #print(trans_test)

        # Load data
        self.train = UnsupervisedMNIST(root, train=True, transform=Compose(pil_transforms), download=download)
        self.train_eval_set = UnsupervisedMNIST(root, train=False, transform=Compose(pil_transforms))
        self.test = UnsupervisedMNIST(root, train=False, transform=Compose(eval_transformation))
