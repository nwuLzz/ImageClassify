# 训练分类模型的主程序
import os
import torch
import torch.nn as nn
import argparse
# 省略了一些模块的引入
from efficientnet_pytorch.model import EfficientNet
from dataset import build_data_set
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def parse_arg():
    parser = argparse.ArgumentParser(description='Args for image classification')     # 创建一个解析对象
    # 向对象中添加参数和选项，用来指定程序需要接受的命令参数
    parser.add_argument('--classes_num', help='类别数', default=4)
    parser.add_argument('--train_data', help='训练文件夹的位置', default='./data/train')
    parser.add_argument('--test_data', help='测试文件夹的位置', default='./data/test')
    parser.add_argument('--image_size', help='输入图片的宽与高，B0推荐224', default=224)
    parser.add_argument('--batch_size', help='batchsize数', default=32)
    parser.add_argument('--workers', help='Dataloader的worker数', default=2)
    parser.add_argument('--epochs', help='epoch数', default=10)
    parser.add_argument('--lr', help='学习率', default=0.001)
    parser.add_argument('--checkpoint_dir', help='模型保存位置', default='./checkpoints')
    parser.add_argument('--save_interval', help='保存间隔，每1个epoch保存一次', default=1)
    parser.add_argument('--momentum', help='momentum动量', default=0.9)
    parser.add_argument('--weight_decay', help='权重衰减', default=1e-04)
    parser.add_argument('--arch', help='使用网络结构', default='efficientnet-b0')
    parser.add_argument('--pretrained', help='是否加载预训练模型', type=bool, default=True)

    args = parser.parse_args()      # 进行解析
    # print(args.train_data)
    return args


def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()

    batch_cnt = 1
    for i, (images, target) in enumerate(train_loader):
        # print("input: ", images)
        # print("target: {}, length: {}".format(target, len(target)))
        # compute output
        output = model(images)              # images为输入，output为模型预测的输出
        # print("output: {}, length: {}".format(output, len(output)))
        loss = criterion(output, target)    # 损失函数
        print('batch: {}    loss:{}'.format(batch_cnt, loss))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_cnt += 1


def model_eva(model_pth):
    """
        模型评估
    :return:
    """
    # 先定义网络结构
    model = EfficientNet.from_name(args.arch)
    # 修改全连接层的神经元数目
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, args.classes_num)
    # 加载保存的参数
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    # for parameter in model.named_parameters():
    #     print(parameter)

    # 用测试集进行评估
    test_dataset = build_data_set(args.image_size, args.test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,      # 一次预测几张照片
        shuffle=False,
        num_workers=args.workers)
    # 分批预测
    y_true = torch.tensor([])   # 初始化所有测试集图片的实际类别
    y_pred = torch.tensor([])   # 初始化所有测试集图片的预测类别
    batch_cnt = 1
    for i, (images, target) in enumerate(test_loader):
        # print("\n测试集第 {} 批预测情况：".format(batch_cnt))
        output = model(images)
        m = nn.Softmax(dim=1)
        output = m(output)      # 利用softmax将预测的每类概率转为0~1之间，所有类的预测概率和为1
        y_pred_batch = output.argmax(dim=1)     # top1预测类别
        # print("预测结果（全）：", output)
        # print("top1预测概率：", output.max(dim=1).values)
        # print("top1预测类别：", y_pred_batch)
        # print("实际类别：", target)
        batch_cnt += 1

        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
        y_true = torch.cat((y_true, target), dim=0)

    print("\n*****************************************")
    print("所有{}张图片的预测类别：{}".format(len(y_pred), y_pred))
    print("所有{}张图片的实际类别：{}".format(len(y_true), y_true))

    # 绘制混淆矩阵
    cm = plot_cm(y_true, y_pred)

    # 计算每类的精确度、召回率
    cal_acc(cm)


def cal_acc(cm):
    """
        基于混淆矩阵计算每类的精确度、召回率
    """
    n = len(cm)
    acc_cnt = 0     # 分类正确的数量
    all_cnt = 0     # 所有样本数
    for c in range(n):
        acc_cnt += cm[c][c]
        all_cnt += sum(cm[c])
    acc_rate = acc_cnt / float(all_cnt)
    print('准确率: ', acc_rate)

    for i in range(len(cm[0])):
        rowsum, colsum = sum(cm[i]), sum(cm[r][i] for r in range(n))    # 每类的实际数量、预测数量
        try:
            print('第 {} 类\t precision:{}\t recall:{}'.format(i, cm[i][i] / float(colsum), cm[i][i] / float(rowsum)))
        except ZeroDivisionError:
            print('第 {} 类\t precision:0\t recall:0'.format(i))


def plot_cm(y_true, y_pred):
    """
    绘制混淆矩阵热力图
    :param y_true: 实际结果 torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    #     3, 3, 3, 3, 3])
    :param y_pred: 预测结果 torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,
    #         1, 1, 1, 1, 1])
    :return:
    """
    sns.set()
    f, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵：\n", cm)
    sns.heatmap(cm, annot=True, ax=ax, cmap='YlOrRd')  # 画热力图

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()

    return cm


def main(args):
    # part1: 模型加载
    # args.classes_num = 4
    args.advprop = False
    if args.pretrained:     # 自动下载并加载 pretrained model 后进行训练
        model = EfficientNet.from_pretrained(args.arch, num_classes=args.classes_num, advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:       # 使用随机数初始化网络
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, override_params={'num_classes': args.classes_num})
    # 有GPU的话，加上cuda()
    # mode.cuda()

    # part2: 损失函数、优化方法
    criterion = nn.CrossEntropyLoss()  # 有GPU的话加上.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_dataset = build_data_set(args.image_size, args.train_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,                  # 传入的数据集
        batch_size=args.batch_size,     # 每批训练几张照片
        shuffle=True,                   # 数据是否打乱
        num_workers=args.workers)       # 进程数，0表示只有主进程

    for epoch in range(args.epochs):
        print('\n\n***********epoch分割线*************')
        print('Epoch {} 训练过程...'.format(epoch))
        # 调用train函数进行训练
        train(train_loader, model, criterion, optimizer, epoch, args)
        # 模型保存
        if epoch % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar.epoch_%s' % epoch))
            # 只保存训练好的参数，state_dict存储的是模型可训练的参数，可打印出来查看
            # print('\n训练好的参数示例：')
            # print(model.state_dict()['_fc.bias'])   # 只打印最后一层的bias


if __name__ == '__main__':
    args = parse_arg()
    # 模型训练
    main(args)

    # 模型评估
    model_pth = os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar.epoch_9')
    model_eva(model_pth)
