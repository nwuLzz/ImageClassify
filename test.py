"""
    类方法和静态方法使用示例
    分类模型构建与评估过程中相关功能测试
"""

from time import time, localtime, sleep
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Clock(object):
    """数字时钟"""

    def __init__(self, hour=0, minute=0, second=0):
        self._hour = hour
        self._minute = minute
        self._second = second

    @classmethod
    def now(cls):
        ctime = localtime(time())
        return cls(ctime.tm_hour, ctime.tm_min, ctime.tm_sec)

    def run(self):
        """走字"""
        self._second += 1
        if self._second == 60:
            self._second = 0
            self._minute += 1
            if self._minute == 60:
                self._minute = 0
                self._hour += 1
                if self._hour == 24:
                    self._hour = 0

    def show(self):
        """显示时间"""
        return '%02d:%02d:%02d' % \
               (self._hour, self._minute, self._second)


def main():
    # 通过类方法创建对象并获取系统时间
    clock = Clock.now()
    while True:
        print(clock.show())
        sleep(1)
        clock.run()


def plot_cm():
    sns.set()
    f, ax = plt.subplots()
    # 混淆矩阵
    y_true = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
        3, 3, 3, 3, 3])
    y_pred = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,
            1, 1, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print(cm)
    # sns.heatmap(cm, annot=True, ax=ax, cmap=plt.get_cmap(name='Reds'))  # 画热力图
    sns.heatmap(cm, annot=True, ax=ax, cmap='YlOrRd')  # 画热力图

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


if __name__ == '__main__':
    # main()
    plot_cm()
