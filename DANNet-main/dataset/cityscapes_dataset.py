import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from dataset.transforms import *
import torchvision.transforms as standard_transforms


# 自定义的 Cityscapes 数据集类，继承自 PyTorch 的 Dataset 类
class cityscapesDataSet(data.Dataset):
    # 构造函数
    def __init__(self, args, root, list_path, max_iters=None, set='val'):
        self.root = root  # 数据集的根目录
        self.list_path = list_path  # 包含图像名的列表文件路径

        # 初始化用于图像输入的变换列表----一系列的数据预处理
        train_input_transform = []
        # 添加变换：将图片转换为张量并进行标准化
        train_input_transform += [standard_transforms.ToTensor(),#将图像转换成tensor
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        # 定义 Cityscapes 数据集特有的变换操作-----
        """这些操作是否是通用的"""
        cityscape_transform_list = [joint_transforms.RandomSizeAndCrop(args.input_size, False, pre_size=None,
                                                                       scale_min=0.5, scale_max=1.0, ignore_index=255),
                                    joint_transforms.Resize(args.input_size),
                                    joint_transforms.RandomHorizontallyFlip()]#随机水平翻转
        # 组合上述定义的变换操作
        self.joint_transform = joint_transforms.Compose(cityscape_transform_list)

        # 定义标签的变换：将分割标签转换为张量   将Mask变成张量图
        self.target_transform = extended_transforms.MaskToTensor()
        # 应用到图像的标准变换操作
        """compose--->将多个变换过程合一"""
        self.transform = standard_transforms.Compose(train_input_transform)#针对图像的一些操作



        # 读取图像列表并进行处理
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # 如果指定了最大迭代次数，调整图像列表以适应这一需求
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        # 初始化文件列表
        self.files = []
        self.set = set  # 指定数据集的子集（如训练、验证等），判断当前数据集的用途
        
        # 整理每张图片的索引和标签
        for name in self.img_ids:
            """img_ids只是索引吗"""
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name)).replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        # 创建一个映射，用于将 Cityscapes 原始标签 ID 映射到训练标签 ID
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # 打印加载的图像数量
        print('{} images are loaded!'.format(len(self.files)))

    """len 和 getitem 是必须要实现的方法"""
    def __len__(self):
        return len(self.files) #成功载入了多少张图片

    """可以索引、迭代数据集"""
    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label = np.asarray(label, np.uint8)

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        label = Image.fromarray(label_copy.astype(np.uint8))

        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        size = image.shape
        return image, label, np.array(size), name


