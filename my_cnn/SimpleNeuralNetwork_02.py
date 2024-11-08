import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.sgd as sgd
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

"""
在01的基础上，体验反向传播的魅力
使用反向传播来调整参数，这次只尝试让他调整final_bias的值

这有个问题，只更新一个参数还好，但是如果要更新两个以上的参数，就不行了，为啥呢
"""
class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b01 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        # 只需要把requires_grad的值变成True就可以让他被训练了
        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b01
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)
        return output

if __name__ == '__main__':

    # 准备训练数据
    train_data = torch.tensor([0.0, 0.5, 1.0])
    train_labels = torch.tensor([0, 1, 0])

    #初始化优化器
    model = BasicNN_train()
    optimizer = sgd.SGD(model.parameters(), lr=0.01)

    print(f"Final bias: before optimization: { model.final_bias } \n")

    num_epochs = 400
    # 开始训练
    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(len(train_data)):
            input_i = train_data[i]
            label_i = train_labels[i]
            # 前向传播
            output_i = model(input_i)
            # 计算损失
            loss = (output_i - label_i) ** 2
            # 反向传播
            loss.backward()
            total_loss += loss

        if total_loss < 0.0001:
            break
        # 更新参数
        optimizer.step()
        # todo 这里不懂为什么要把梯度清零，等我回去看了视频再研究一下
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")


    input_doses = torch.linspace(start=0, end=1, steps=11)

    output_values = model(input_doses)

    sns.set_theme(style="whitegrid")
    sns.lineplot(x=input_doses, y=output_values.detach(), color="blue", linewidth=2)

    plt.ylabel("Effectiveness")
    plt.xlabel("Dose")
    plt.show()