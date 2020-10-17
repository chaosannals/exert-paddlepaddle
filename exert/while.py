from paddle import fluid
from paddle.fluid import layers
import numpy

# 条件体，参数对应 loop_vars
def cond(i, ten):
    # print('cond: {} < {}'.format(i, ten))
    return i < ten

# 循环体，参数对应 loop_vars
def body(i, ten):
    i = i + 3
    return i, ten # 作为 结果 或 下次循环的参数

# 定义常量
i = layers.fill_constant(shape=[1], value=0, dtype='int64')
ten = layers.fill_constant(shape=[1], value=10, dtype='int64')

# 定义循环体 loop_vars 首次传入的参数
r_i, r_ten = layers.while_loop(cond=cond, body=body, loop_vars=[i, ten])

print(r_i)
print(r_ten)

# 网络参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
result = exe.run(fluid.default_main_program(), feed={}, fetch_list=[r_i, r_ten])

print(result)
