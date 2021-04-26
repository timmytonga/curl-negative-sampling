import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm
import logging

import utils
from utils import set_seed, log_args
from model import Model
import pickle
import os

logger = logging.Logger(__name__)


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, feature_dim=128, model_select="resnet50", width=64):
        super(Net, self).__init__()

        # encoder
        model = Model(feature_dim, model_select, width).cuda()

        model.load_state_dict(torch.load(pretrained_path))
        proj_in = width*8
        self.f = model.f
        # classifier
        self.fc = nn.Linear(proj_in, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model', choices=['resnet10vw', 'resnet50'], default='resnet50')
    parser.add_argument('--width', default=None, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--logpath', default="linear.log", type=str)
    parser.add_argument('--result_dir', default='linear_results', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--parallelize', default=False, action='store_true', help='Call nn.DataParallel on model or not')
    parser.add_argument('--num_neg', default=None, type=int)

    args = parser.parse_args()
    assert args.num_neg is not None
    if args.model == 'resnet10vw':
        assert args.width is not None
    set_seed(args.seed)
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs

    log_level = logging.INFO if args.verbose else logging.DEBUG
    logger = utils.get_logger(name=__name__, filename=args.logpath, console_log_level=log_level)  # default we log everything to console
    log_args(args, logger)

    logger.info("Loading Data")
    train_data = STL10(root='data', split='train', transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_data = STL10(root='data', split='test', transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("Loading Model")
    model = Net(num_class=len(train_data.classes), pretrained_path=model_path,
                model_select=args.model, width=args.width).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    if args.parallelize:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    logger.info("TRAINING")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        # logger.info(f"[EPOCH {epoch}] Train_loss: {train_loss}, train_acc_1: {train_acc_1}, train_acc_5: {train_acc_5}")
        # logger.info(f"[EPOCH {epoch}] test_loss: {test_loss}, train_acc_1: {test_acc_1}, train_acc_5: {test_acc_5}")
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    pickle.dump(results, open(os.path.join(args.result_dir, f"linearN{args.num_neg}.p"), "wb"))
