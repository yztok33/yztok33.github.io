## PU Learning

### 一、背景介绍

在现实生活中，有许多只有正样本和大量未标记样本的例子。这是因为负类样本的一些特点使得获取负样本较为困难。比如：

1. 负类数据不易获取。
2. 负类数据太过多样化。
3. 负类数据动态变化。

比如在推荐系统里，只有用户点击的正样本，却没有**显性**负样本，我们不能因为用户没有点击这个样本就认为它是负样本，因为有可能这个样本的位置很偏，导致用户没有点。

PU Learning（Positive-unlabeled learning）是半监督学习的一个研究方向，指在只有正类和无标记数据的情况下，训练二分类器，伊利诺伊大学芝加哥分校（UIC）的刘兵（Bing Liu）教授和日本理化研究所的杉山将（Masashi Sugiyama）实验室对PU Learning有较深的研究。

应用场景：恶意URL检测、致病基因检测等

### 二、方法介绍

目前有两种解决方法：

1、启发式地从未标注样本里找到可靠的负样本，以此训练二分类器，该方法问题是分类效果严重依赖先验知识。

2、将未标注样本作为负样本训练分类器，由于负样本中含有正样本，错误的标签指定导致分类错误。

#### 2.1 直接利用标准分类方法

将正样本和未标记样本分别看作是positive samples和negative samples, 然后利用这些数据训练一个标准分类器。分类器将为每个物品打一个分数（概率值），通常正样本分数高于负样本的分数，因此对于那些未标记的物品，分数较高的最有可能为positive。（自己设定比例或者阈值进行划分）

#### 2.2 PU bagging

a)通过将所有正样本和未标记样本进行随机组合来创建训练集；

b)利用这个“bootstrap”样本来构建分类器，分别将正样本和未标记样本视为positive和negative；

c)将分类器应用于不在训练集中的未标记样本 - OOB（“out of bag”）- 并记录其分数；

d)重复上述三个步骤，最后为每个样本的分数为OOB分数的平均值。

这是一种bootstrap的方法，可以理解为之前我们会想到**随机抽取一部分未标记样本U作为负样本**来训练，在这里会设置迭代次数T，根据正样本的个数，每次都随机可重复地从U中选取和P数量相同的样本作为负样本N，并打上标签，每次迭代都重复进行取样->建模->预测的过程，最后的预测概率使用T次迭代的平均值作为最终预测的概率。

#### 2.3 Two-step approaches

大部分的PU learning策略属于“two-step approaches”。最近的一篇介绍这些方法的论文是 An Evaluation of Two-Step Techniques for Positive-Unlabeled Learning in Text Classification。

a)识别可以百分之百标记为negative的未标记样本子集（“reliable negatives”）；需要较大的人工标注

b)使用正负样本训练标准分类器并将其应用于剩余的未标记样本。

通常，会将第二步的结果返回到第一步并重复上述步骤。即每一轮循环都会找出那些所谓百分之百的正样本和负样本，加入到训练集里，重新预测剩余的未标记样本，直到满足停止条件为止。

#### 2.4 Positive unlabeled random forest

这里值得一提的关于PU learning的最新一个发展是文献Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework中提出的一种算法。

所提议的框架，称为PURF（正无标签随机森林），能够从正面和未标记实例中学习，通过并行计算根据UCI数据集上的实验，通过完全标记数据训练的RF实现可比较的分类性能。该框架将包括广泛使用的PU信息增益（PURF-IG）和新开发的PU基尼指数（PURF-GI）的PU学习技术与可扩展的并行计算算法（即RF）相结合。

**并行化步骤：**

1、创建t棵树、4个进程，每个进程负责创建t/4棵决策树，创建好的t/4棵决策树以列表形式返回主进程；

2、分别得到4个子进程的决策树列表后，将4个子列表整合到一个长度为t的决策树列表L；

3、创建4个分类进程，将决策树列表复制4份分别传递到4个分类进程，同时将测试数据分成4份，[0,388]行为第1部分，[389,777]行为第2部分，[778,1166]行为第3部分，[1167,1558]行为第4部分，分别传递到4个分类子进程；

4、第一个子进程以列表的形式返回[0,388]行的分类结果，第二个子进程以列表的形式返回[389,777]行的分类结果，第三个子进程以列表的形式返回[778,1166]行的分类结果，第四个子进程以列表的形式返回[1167,1558]行的分类结果。

5、分别得到4个子进程的标签列表之后，将4个子列表整合到一个长度为1559的结果标签列表。

#### 2.5 参考代码（介绍）

[https://github.com/phuijse/bagging_pu/blob/master/PU_Learning_simple_example.ipynb](https://link.zhihu.com/?target=https%3A//github.com/phuijse/bagging_pu/blob/master/PU_Learning_simple_example.ipynb)（PU_Learning_simple_example.ipynb）

[https://github.com/roywright/pu_learning/blob/master/circles.ipynb](https://link.zhihu.com/?target=https%3A//github.com/roywright/pu_learning/blob/master/circles.ipynb)（PU learning techniques applied to artificial data“circle”）

#### 2.6 实战例子：

囿于篇幅，我们从中挑选出一个例子，进行介绍：

人工构造了Circles数据集，如下图所示：

![CSDN图标](https://i.loli.net/2020/08/29/iIXSsGxKNQdpwYj.png)

上图一共有6000个样本点，真实得正样本和负样本均为3000个，只不过，我们只知道其中**300**个正样本，剩余的5700个样本认为是unlabeled样本。在该数据集上，分别应用以上3种方法，结果分别为:

![CSDN图标](https://i.loli.net/2020/08/29/Wespogn7Y6bvu9V.png)

![CSDN图标](https://i.loli.net/2020/08/29/EfNejyKzLA3I6bs.png)

![CSDN图标](https://i.loli.net/2020/08/29/4BEVmAJYhQn3tFN.png)

我们对比一下3种方法的性能，这里的性能指的是：对于预测的样本（5700个），依次取前100，200，300直到2700个（剩余的真的正样本的个数）样本，看下取出的这些样本真正是正样本的概率。（看不明白的，可以详细看下代码）。

![CSDN图标](https://i.loli.net/2020/08/29/DmQiEfxGzMHLsZy.png)

图中Average score是3种方法的平均。可以看出来，在有300个正样本的Circles数据集上，PU bagging的方法最好。

------

根据参考文献【1】的所有实验，我总结出以下的结果，详细请参考原文：

注：所有的数据集都是6000个样本，2类，每一类为3000个。我们已知的正样本的数目为hidden_size。

①对于Circles数据集：

- hidden_size为1000时，Standard方法最好，PU bagging最差。
- hidden_size为300时，PU bagging方法最好，Standard最差。
- hidden_size为30时，PU bagging方法最好，Standard最差。

②对于Two moon数据集：

- hidden_size为1000时，Standard方法最好，PU bagging最差。
- hidden_size为300时，PU bagging方法最好，Standard最差。
- hidden_size为30时，PU bagging方法最好，Standard最差。

③对于Blobs数据集：

- hidden_size为1000时，Standard方法最好，PU bagging最差。
- hidden_size为300时，PU bagging方法最好，Standard最差。
- hidden_size为30时，PU bagging方法最好，Standard最差。

④对于PU bagging方法：
决策树作为基分类器的效果比起SVM作为基分类器的效果差。

通过上述的结果，和各个方法的理论，是否可以大胆做出一个结论呢？即**随着已知正样本比例的减少，PU bagging最好，Standard最差，两步法居中**。如果我们的正样本的比例只占全部样本的很小的部分，根据上述的结论，应该选用PU bagging策略。

### 三、Estimating the Class Prior in Positive and Unlabeled Data through Decision Tree Induction（类先验）

论文通过决策树归纳对数据子域概率给出下限，随着标记示例比率的增加，该下限更接近实际概率。论文方法的估计与现有技术方法的估计一样准确，并且速度提高了一个数量级。

#### 3.1 应用背景

1、医疗记录通常只列出每个人的诊断疾病，而不是该人没有的疾病，没有诊断并不意味着患者没有患病；

2、知识库（KB）完成的任务本质上也是一个积极且无标签的问题，自动构造的KB只包含真实的事实，并不完整，未包括在KB中的事实的真值是未知的，但并不一定错误；

3、文本分类也可通过正样本和未标记数据来表征，如对用户的网页首选项进行分类可以将带书签的页面用作正例，将所有其他页面用作未标记的页面。

#### 3.2 方法介绍

知道标签频率c（为正样本或副样本）大大简化了PU学习。首先，可以训练概率分类器来预测Pr，并调整输出概率；其次，使用相同的分类器对未标记的数据进行加权，然后对加权数据训练不同的分类器。第三，使用下列等式修改学习算法，如基于计数的算法——树归纳和朴素贝叶斯，只考虑数据的属性条件子集中正例和负例的数量。标签频率可通过三种方式获得：来自领域知识、通过从小的完全标记数据集估计、直接根据PU数据估算。

![img](https://i.loli.net/2020/08/29/tV2On4Zs6XlopUa.jpg)

论文提出了一种简单有效的方法估计类先验，该方法基于以下观点：标签频率预期在属性的任何子域中相同，数据的子集自然地暗示标签频率的下限。使用基于PU数据的决策树归纳可以容易地找到可能的正子域。论文将以下先前估计方法进行比较，使用了“完全随机选择”假设：EN（Elkan和Noto 2008），PE（du Plessis和Sugiyama 2014），pen-L1（du Plessis，Niu和Sugiyama 2015），KM1和KM2（Ramaswamy，Scott和Tewari 2016），AlphaMax（Jain等人2016）和AlphaMax N（Jain，White和Radivojac 2016）。与这些论文的作者一样，本文对数据集二次抽样，最多包含2000个示例，并重复该过程五次。

论文目标是深入了解TIcE（Tree Induction for Label Frequency Estimation）的性能，用于c估计的树诱导，估计来自PU数据的标签频率。首先，检查在实践中是否最好采用下限的最大值或使用一个下限；其次，评估设置δ的方法；最后，将TIcE与其他类先验估计算法进行比较。

该算法将数据集分成两个独立的集合，使用一组可能是正样本的子域，并使用另一个集合通过最紧密下限来估计c在子域中的计算。寻找数据中纯子集也是决策树归纳的目标，因此TIcE通过引入决策树来寻找纯标记子集，将未标记数据视为负数。

拆分标准决策树归纳的目标是找到纯节点，使用阳性比例（max-bepp）得分的最大偏差估计值，选择给出具有最高bepp的子集的分裂：TP。

## 参考文献

1-Learning from Positive and Unlabeled Examples with Different Data Distributions

2-Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework

3-Positive-Unlabeled Learning with Non-Negative Risk Estimator

4-Estimating Rule Quality for Knowledge Base Completion with the Relationship between Coverage Assumption

5-Beyond the Selected Completely At Random Assumption for Learning from Positive and Unlabeled Data

6-Learning From Positive and Unlabeled Data: A Survey

7-https://zhuanlan.zhihu.com/p/82556263

8-https://blog.csdn.net/anshuai_aw1/article/details/89475986