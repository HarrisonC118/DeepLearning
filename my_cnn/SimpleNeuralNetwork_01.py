import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.sgd as sgd
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

"""
这个代码是想实现一个全手动的神经网络，层数没有很多，参数全靠手写
主要内容是：吃药，用0表示不吃，1表示全吃。不吃和全吃对病都不好。只有吃一半才能治好这个病
设计一个神经网络让他自动预测吃药有效的概率
"""
class BasicNN(nn.Module):
    # 这个类的作用是初始化参数，也就是权重和偏置，在创建这个类时，就会自动调用这个类的__init__方法
    def __init__(self):
        super().__init__()
        # nn.Parameter()是torch里面的一种变量形式，这个类可以看成是一种特殊的变量，可以在反向传播时被pytorch自动更新
        # requires_grad是requires_gradient的缩写，gradient的意思是梯度，也就是偏导数
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b01 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        # scaled的意思是缩放，todo 暂时还不知道这段代码有什么含义，等补完前面的课程再说
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b01
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)
        return output

if __name__ == '__main__':
    # 测试是否有用

    # 这段代码的作用是生成一个等差数列
    input_doses = torch.linspace(start=0, end=1, steps=11)
    model = BasicNN()
    output_values = model(input_doses)

    # 画出一个漂亮的图表
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=input_doses, y=output_values, color="blue", linewidth=2)

    plt.ylabel("Effectiveness")
    plt.xlabel("Dose")
    plt.show()