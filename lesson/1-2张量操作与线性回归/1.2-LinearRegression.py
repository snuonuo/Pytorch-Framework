import torch
import numpy as np
import matplotlib.pyplot as plt

# 初始化超参数
lr = 0.05
# 创建数据
x = torch.rand(20, 1) * 10
y = 2*x + 5 + torch.randn(20, 1)

# 创建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.randn((1), requires_grad=True)

for iter in range(1000):
    # 向前传播
    wx = torch.mul(w,x)
    y_pred = torch.add(wx, b)

    # 计算loss
    loss = (0.5 * (y - y_pred)**2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data = b.data - lr * b.grad
    w.data = w.data - lr * w.grad

    # 参数梯度清零
    w.grad.zero_()
    b.grad.zero_()

    # 绘图代码
    if iter % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iter, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break



