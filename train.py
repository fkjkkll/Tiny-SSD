""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 14:00:27
"""

from tqdm import tqdm
from utils.tools import *
from utils.dataLoader import MyDataSet, dataset_collate
from torch.utils.data import DataLoader
from model.anchor_generate import generate_anchors
from model.anchor_match import multibox_target
from model.net import TinySSD
from model.loss import *

# -----------------------------------------------
#                   产生先验锚框
# -----------------------------------------------
sizes = get_anchor_info('model_data/anchor_sizes.txt')
ratios = get_anchor_info('model_data/anchor_ratios.txt')
if len(sizes) != len(ratios): ratios = [ratios[0]] * len(sizes)
anchors_perpixel = len(sizes[0]) + len(ratios[0]) - 1
feature_map = [32,16,8,4,1]
anchors = generate_anchors(feature_map, sizes, ratios)

# -----------------------------------------------
#                   加载数据
# -----------------------------------------------
input_shape = 256
_, num_classes = get_classes('model_data/voc_classes.txt')
with open('2077_train.txt') as f:
    train_lines = f.readlines()
train_dataset = MyDataSet(train_lines, input_shape, mode='train')
train_iter = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          pin_memory=True, drop_last=True, collate_fn=dataset_collate)

# -----------------------------------------------
#                   网络部分
# -----------------------------------------------
device, net = try_gpu(), TinySSD(anchors_perpixel=anchors_perpixel, num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, 10, 0.0001)

# -----------------------------------------------
#                   开始训练
# -----------------------------------------------
num_epochs, timer = 30, Timer()
timer.start()
animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
net = net.to(device)
anchors = anchors.to(device)
cls_loss, bbox_loss = None, None
for epoch in range(num_epochs):
    print(f'learning rate: {scheduler_lr.get_last_lr()}')
    metric = Accumulator(4)
    net.train()
    for features, target in tqdm(train_iter):
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device) # (bs, 3, 256, 256) (bs, 100, 5)

        # 为每个锚框预测类别和偏移量(多尺度结果合并)
        cls_preds, bbox_preds = net(X) # (bs, 5444, 2) (bs, 5444*4)
        
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y) # (bs, 5444*4) (bs, 5444*4) (bs, 5444)

        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), 1, bbox_eval(bbox_preds, bbox_labels, bbox_masks), 1)

    scheduler_lr.step()
    cls_loss, bbox_loss = metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_loss, bbox_loss))
    print(f'epoch {epoch+1}/{num_epochs}: ', 'cls-loss: ',metric[0] / metric[1], ' box-loss', metric[2] / metric[3])

    # 每个epoch都保存训练模型
    torch.save(net.state_dict(), f'model_data/result_{epoch+1}.pt')

print(f'class loss {cls_loss:.2e}, bbox loss {bbox_loss:.2e}')
print(f'total time: {timer.stop():.1f}s', f' on {str(device)}')















