#### 第一问：

由题意，建立模型对球员表现进行评估。

我们确立如下积分规则：

> 初始积分为0分，当前发球球员+5分，发球球员第二次发球-2分，赢球+1
> ace+3分，对方ace-2分，winner+2分，对方winner-1分，反手得分+1分，双误-3分，对方双误+2分，非受迫性失误-4分，对方非受迫性失误+3分，网前+2分，对方网前-1分，破发点+2分，对方破发点-1分，面临破发点-1分，对手面临破发点-2分，未赢得破发点-1分，对手未赢得破发点+1分
> 总击打数：0-5次不加分、6-10 +1分、10-20 +2分、20+ +3分
> 击球速度：0-5次不加分、6-10 +1分、10-20 +2分、20+ +3分
> 发球宽度：C+0分 BC+1分 B+2分 BW+3分 W+4分 NA-1分
> 发球深度：CTL+3分 NCTL+2分 NA-1分
> 回球深度：D+3分 ND+2分 NA-1分
>
> 双方有分差-1分
> 大场分低-3分，小场分低-2分，当局分数低-1分

我们通过自己定义的公式对每位选手的每个时刻进行打分：

经过计算和统计，所有选手的得分值在-12到24之间：

我们将得到的选手结果进行可视化：其中选手1的表现分大于选手2的点为红色，选手2的表现分大于选手1的点为蓝色。

<img src="C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20240204005357777.png" alt="image-20240204005357777" style="zoom:50%;" />

计算每个时刻的表现分差值：     S=S1-S2

绘制柱状图，对S与实际的得分结果进行可视化

<img src="C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20240204010508424.png" alt="image-20240204010508424" style="zoom: 33%;" />

单独取出最后一场比赛的选手双方的表现分，画出随时间序列的分布图。

<img src="C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20240204011831297.png" alt="image-20240204011831297" style="zoom:50%;" /><img src="C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20240204012333251.png" alt="image-20240204012333251" style="zoom:50%;" />

由图可知，德约科维奇在决赛中的发挥表现很差，也预示着他输掉这场比赛。

（还可以画其他比赛的图）

#### 第二问：

要反驳教练的观点，即证明momentum在比赛中是发挥了作用的。

我们首先对我们第一问得到的模型对每一分的得分情况进行分析，得到了，接着进行相关性分析。

放**Pearson相关系数、Spearman秩相关系数**的原理公式

计算`final_score1`、`final_score2`、`score1-2` 与 `point_victor` 列的相关性

即选手1、选手2、得分差值与真实值的关系

Pearson Correlation Coefficient (final_score1): -0.6420179463897615
Spearman Correlation Coefficient (final_score1): -0.6577805524913795
Pearson Correlation Coefficient (final_score2): 0.6355928335630012
Spearman Correlation Coefficient (final_score2): 0.6470547713565329
Pearson Correlation Coefficient (score1-2): -0.7001305114137116
Spearman Correlation Coefficient (score1-2): -0.7115973930837495

均大于0.5，所以不是随机的。

对此，绘制热力图：

<img src="C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20240204014024734.png" alt="image-20240204014024734" style="zoom:50%;" />

这部分放原来的论文，求得势头。

第三问：

确定指标对表现的影响：即分析各指标的权重值