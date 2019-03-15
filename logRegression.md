# logistic回归

## 1.定义<br>
&emsp;&emsp;logistic回归是处理二分类问题的，其输出y={0, 1}。输入**X**是一个m维的样本特征向量，所以其输入输出关系可按单位阶跃函数表示：<br>

$$y=\begin{cases}
0 & z<0 \\
0.5 & z=0 \\
1 & z>0
\end{cases}$$

但是单位阶跃函数是非连续的，这里使用sigmoid函数来取代单位阶跃函数<br>

y=\frac{1}{1+e^{-z}}

在原线性回归模型上加上sigmoid函数，便形成了logistic回归的预测函数，可以用于二分类问题：

y=\frac{1}{1+e^{-(-W^{T}X+B)}}

具体可见https://blog.csdn.net/feilong_csdn/article/details/64128443
