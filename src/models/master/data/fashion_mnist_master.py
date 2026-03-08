#from survae.model.data.datasets.image import UnsupervisedMNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose

from survae.model.data import TrainTestLoader, DATA_PATH


class Fashion_MNIST(TrainTestLoader):
    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], eval_transformation=[]):

        self.root = root

        # Load data
        self.train = FashionMNIST(root, train=True, transform=Compose(pil_transforms), download=download)
        #self.train_eval_set = FashionMNIST(root, train=False, transform=Compose(pil_transforms))
        self.test = FashionMNIST(root, train=False, transform=Compose(eval_transformation))
