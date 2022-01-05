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
#                   配置信息
# -----------------------------------------------
voc_classes_path = 'model_data/voc_classes.txt'
image_size_path = 'model_data/image_size.txt'
train_file_path = '2077_trainval.txt'
anchor_sizes_path = 'model_data/anchor_sizes.txt'
anchor_ratios_path = 'model_data/anchor_ratios.txt'

# -----------------------------------------------
#                   加载数据
# -----------------------------------------------
_, num_classes = get_classes(voc_classes_path)
r = get_image_size(image_size_path)
with open(train_file_path) as f:
    train_lines = f.readlines()
train_dataset = MyDataSet(train_lines, r, mode='train')
train_iter = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          pin_memory=True, drop_last=True, collate_fn=dataset_collate)


# -----------------------------------------------
#                   产生先验锚框
# -----------------------------------------------
sizes = get_anchor_info(anchor_sizes_path)
ratios = get_anchor_info(anchor_ratios_path)
if len(sizes) != len(ratios): ratios = [ratios[0]] * len(sizes)
anchors_perpixel = len(sizes[0]) + len(ratios[0]) - 1
feature_map = [r // 8, r // 16, r // 32, r // 64, 1]
anchors = generate_anchors(feature_map, sizes, ratios) # (1600+400+100+25+1)*4个锚框


# -----------------------------------------------
#                    网络部分
# -----------------------------------------------
device, net = try_gpu(), TinySSD(app=anchors_perpixel, cn=num_classes)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, 200)

# -----------------------------------------------
#                    开始训练
# -----------------------------------------------
num_epochs, timer = 1000, Timer()
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
        X, Y = features.to(device), target.to(device) # (bs, 3, h, w) (bs, 100, 5)

        # 为每个锚框预测类别和偏移量(多尺度结果合并)
        cls_preds, bbox_preds = net(X) # (bs, anchors, (1+c)) (bs, anchors*4)
        
        # 为每个锚框标注类别和偏移量 (bs, anchors*4) (bs, anchors*4) (bs, anchors)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), 1, bbox_eval(bbox_preds, bbox_labels, bbox_masks), 1)

    # 学习率衰减
    scheduler_lr.step()

    # 留作显示
    cls_loss, bbox_loss = metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_loss, bbox_loss))
    print(f'epoch {epoch+1}/{num_epochs}: ', 'cls-loss: ', cls_loss, ' box-loss', bbox_loss)

    # 每个epoch都保存训练模型
    torch.save(net.state_dict(), f'model_data/result_{epoch+1}.pt')

print(f'class loss {cls_loss:.2e}, bbox loss {bbox_loss:.2e}')
print(f'total time: {timer.stop():.1f}s', f' on {str(device)}')















