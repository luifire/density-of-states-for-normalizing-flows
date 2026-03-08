from master.data.sphere_data import VonMises
from master.data.line_data import LineData
from master.data.gauss_data import GaussData
from master.data.smoothstep_data import SmoothstepData
from misc.constants import EMBEDDED_DATA


class TrainClassWrapper:
    """Imitates an expected training class / one which contains a train and a test set"""
    TEST_SET_SIZE = 512
    def __init__(self, dataset_name, **kwargs):
        name_to_dataset = {'sphere': VonMises, 'line': LineData,
                           'smooth_step': SmoothstepData, 'gauss': GaussData}
        data_class = name_to_dataset[dataset_name]

        train = data_class(sample_seed=data_class.train_sample_seed, **kwargs)

        #kwargs['size'] = TrainClassWrapper.TEST_SET_SIZE
        test = data_class(sample_seed=data_class.test_sample_seed, **kwargs)

        self.train = train.data[EMBEDDED_DATA]
        self.test = test.data[EMBEDDED_DATA]
        self.dataset_creator = train
