# Reinforcement Learning: An Introduction book 

## Part I: Tabular Solution Methods

表格式解决方法。即状态和动作在很小的一个范围内，可以直接使用矩阵或者表格表示，这里找到的都是最优值。

接下来主要介绍的是有限马尔科夫决策过程，核心是贝尔曼方程和值函数。解决这个问题有三种基础方法：

- 动态规划（dynamic programming）：数学方便实现；需要环境精确认知模型
- 蒙特卡洛搜索（Monte Carlo methods）：无模型；不适用迭代计算
- 时间差分学习（temporal-diﬀerence learning）：无模型，可迭代；过于复杂

剩下两章，主要解决如何将这三种方法的优点结合起来，使用一个组合方法。

## Part II: Approximate Solution Methods

在应对状态空间十分大的问题时，寻找近似解是最好的办法。

主要问题是如何对状态进行泛化（generalization）。一种泛化的方法称作函数逼近，即对值函数进行拟合。这也是一种监督学习的方法。

但函数逼近对传统强化学习而言仍然具有一些新问题，如不稳定问题、自引导、延迟目标等。

## Part III: Looking Deeper

这一部分主要结合神经科学探索强化学习新的研究方向，包括深度强化学习等。