'''
之前的similarity都算是錯了!
沒有完整的follow pFedGraph
真正的pFedGraph會從cosine matrix再去解optimization problem

'''

from scipy.optimize import linprog

# 给定向量 X
X = [1, 0.86, 0.24, 0.23, 0.18, 0.33, 0.2, 0.4, 0.36, 0.44]

# 目标函数系数（即向量 X 的相反数，因为我们要最大化 W · X）
c = [-x for x in X]

# 约束条件矩阵和向量
A_eq = [[1] * len(X)]
b_eq = [1]

# 每个变量的下界为 0
bounds = [(0, None)] * len(X)

# 求解线性规划问题
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# 输出结果
W = res.x
print("W =", W)
print("Maximum dot product =", -res.fun)
