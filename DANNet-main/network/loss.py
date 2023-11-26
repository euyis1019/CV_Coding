import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(index, classes):
    # 定义 one-hot 张量的大小，基于索引大小和类别数
    size = index.size()[:1] + (classes,)
    # 定义一个用于重塑索引张量的视图
    view = index.size()[:1] + (1,)
    # 创建一个零填充的张量，并将其移动到 GPU
    mask = torch.Tensor(size).fill_(0).cuda()
    # 根据定义的视图重塑索引张量
    index = index.view(view)
    # 定义在 one-hot 张量中填充的值
    ones = 1.
    # 在指定索引处将 one-hot 张量填充为一
    return mask.scatter_(1, index, ones)

class StaticLoss(nn.Module):
    def __init__(self, num_classes=19, gamma=1.0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        # 初始化父类（nn.Module）
        super(StaticLoss, self).__init__()
        # 初始化用于损失计算的各种参数
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight
        self.raw = False
        # 如果类别数少于19，设置 raw 标志为真
        if (num_classes < 19):
            self.raw = True

    def forward(self, input, target, eps=1e-5):
        # 获取输入张量的形状
        B, C, H, W = input.size()
        # 重塑并重新排序输入张量的维度
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = （P, C）

        # 如果 raw 为真，创建目标张量的偏移版本
        if self.raw:
            # 初始化偏移目标
            target_left, target_right, target_up, target_down = target, target, target, target
            # 向不同方向进行偏移
            target_left[:, :-1, :] = target[:, 1:, :]
            target_right[:, 1:, :] = target[:, :-1, :]
            target_up[:, :, 1:] = target[:, :, :-1]
            target_down[:, :, :-1] = target[:, :, 1:]
            # 扁平化偏移目标
            target_left, target_right, target_up, target_down = target_left.view(-1), target_right.view(-1), target_up.view(-1), target_down.view(-1)
            # 重复上述步骤，使用不同的偏移集
            target_left2, target_right2, target_up2, target_down2 = target, target, target, target
            target_left2[:, :-1, 1:] = target[:, 1:, :-1]
            target_right2[:, 1:, 1:] = target[:, :-1, :-1]
            target_up2[:, 1:, :-1] = target[:, :-1, 1:]
            target_down2[:, :-1, :-1] = target[:, 1:, 1:]
            # 扁平化第二套偏移目标
            target_left2, target_right2, target_up2, target_down2 = target_left2.view(-1), target_right2.view(-1), target_up2.view(-1), target_down2.view(-1)

        # 扁平化原始目标张量
        target = target.view(-1)
        # 如果 ignore 不是 None，从输入和目标张量中过滤掉忽略的索引
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            # 如果 raw 为真，对偏移目标应用相同的过滤
            if self.raw:
                target_left, target_right, target_up, target_down = target_left[valid], target_right[valid], target_up[valid], target_down[valid]
                target_left2, target_right2, target_up2, target_down2 = target_left2[valid], target_right2[valid], target_up2[valid], target_down2[valid]

        # 如果 one_hot 为真，将目标转换为 one-hot 编码
        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))
            # 如果 raw 为真，对偏移目标执行 one-hot 编码
            if self.raw:
                target_onehot2 = one_hot(target_left, input.size(1)) + one_hot(target_right, input.size(1)) \
                                + one_hot(target_up, input.size(1)) + one_hot(target_down, input.size(1)) \
                                + one_hot(target_left2, input.size(1)) + one_hot(target_right2, input.size(1)) \
                                + one_hot(target_up2, input.size(1)) + one_hot(target_down2, input.size(1))
                target_onehot = target_onehot + target_onehot2
                # 如果 one-hot 编码超过1，则设置为1
                target_onehot[target_onehot > 1] = 1

        # 使用 softmax 计算概率分布
        probs = F.softmax(input, dim=1)
        # 根据 one-hot 编码和权重调整概率分布
        probs = (self.weights * probs * target_onehot).max(1)[0]
        # 将概率限制在 eps 和 1-eps 之间
        probs = probs.clamp(self.eps, 1. - self.eps)
        # 计算对数概率
        log_p = probs.log()

        # 根据公式计算批次损失
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # 根据 size_average 参数决定是计算平均损失还是总损失
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
