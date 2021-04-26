from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
import cv2
import numpy as np
import torch, os

import logging


def log_args(args, logger):
    for argname, argval in vars(args).items():
        if argval is not None:
            logger.info(f'{argname.replace("_"," ").capitalize()}: {argval}')
    logger.info('\n')


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(name: str,
                 filename: str,
                 format: str="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
                 file_log_level: int=logging.INFO,
                 output_to_console: bool=True,
                 console_log_level: int=logging.INFO) -> logging.Logger:
    """
    Return a logger configured to output to file and/or stdout
    :param name: the name of the logger -- usualle __name__ for module name
    :param filename: the name for the log file for the logger to output to ex. "run2.log"
    :param format: a format string according to logging.Formatter (see Python docs for more info)
    :param file_log_level: logging level to log to the file
    :param output_to_console: whether we want this logger to output to stdout or not
    :param console_log_level: what level this logger should output to stdout
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(console_log_level)

    formatter = logging.Formatter(format)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if output_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


np.random.seed(0)


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
