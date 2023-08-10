AlexNet 在2012年提出，以较为简单的结果取得了当年图像识别比赛中傲人的成绩

<img width="1114" alt="image" src="https://github.com/DishengLL/CV/assets/39432361/4982917b-989e-4c4e-b035-8b97f29fa3a6">

## 结构：
- [ ] 5个卷积层
- [ ] 3个全连接层
- [ ] 使用Relu激活函数 （easier to train than `sigmoid` and `tanh` activate function）
- [ ] dropout layer (0.5, in the first two fully connected layers)
- [ ] 针对certain layer的输出使用`local response normalization`
      > This sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons, creating competition for big activities amongst neuron outputs computed using different kernels.
      根据论文， AlexNet在开始的两个Conv的输出上使用了 `local response normalization`
- [ ] overlapping pooling (在特定的pooling layer中使用)

## 工程
- [ ] 使用2个GPU分布式训练
- [ ] 两个GPU只在特定的layer进行通信

## Loss Function
$$
x L(W) = \sum^{N}_{i=1}\sum^{1000}_{C=1} -y_{ic}\log{f_{c}(x_i)} + \epsilon {||w||^{2}_{2}}
\\
N = the \space number \space of \space samples
\\
C = category
$$

### reference 
[original paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  
[Explanation PowerPoint](https://cvml.ista.ac.at/courses/DLWT_W17/material/AlexNet.pdf)

