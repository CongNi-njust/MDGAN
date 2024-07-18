from torch.nn.functional import pad
import numpy as np
from PIL import Image
import glob
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import torchvision.transforms as transforms

class get_data(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.source_list, self.target_list = self._load_dataset()
        with open('D:/.../d.txt', 'r') as f:
            self.numbers = [float(line.strip()) for line in f.readlines()]
        # self.source_list = self._load_dataset()
        self.totensor = torchvision.transforms.ToTensor()

    def _load_dataset(self):
        # glob模块可以获得图像路径
        source_paths = glob.glob(self.args.train_source_dir + "*.png")
        target_paths = glob.glob(self.args.train_target_dir + "*.png")

        return source_paths, target_paths
        # return source_paths

    def __len__(self):
        # return 400
        return len(self.source_list)

    def __getitem__(self, index):

        source_path = self.source_list[index]
        source = Image.open(source_path)
        source = np.array(source)
        source = self.totensor(source).float()

        target_path = self.target_list[index]
        target = Image.open(target_path)
        target = np.array(target)
        target = self.totensor(target).float()
        # target = pad(target, pad=((1024 - 512) // 2, (1024 - 512) // 2, (1024 - 512) // 2, (1024 - 512) // 2),
        #              mode="constant")
        return source, target, torch.tensor(self.numbers[index])
        # return source


def get_dataloaders(args):

    dataset = get_data(args)  #dataset = (source,target)
    data_loader = None

    if len(dataset):

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,  # True 每个epoch重新洗牌数据
            drop_last=True,  # 删除最后一个不满足5组图像的批
        )

    return data_loader