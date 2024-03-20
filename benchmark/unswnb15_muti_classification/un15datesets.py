from torch.utils.data import Dataset
import numpy as np

class UN15Dataset(Dataset):
    def __init__(self, data, label, transform=None):
        """
        纸币分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # self.label_name = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9 }
        self.transform = transform
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        # label = torch.LongTensor([label])
        return data, label

    def __len__(self):
        # return len(self.)
        return len(self.data)

