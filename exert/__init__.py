from paddle import fluid
from paddle.fluid import layers
import numpy

# data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')

a = fluid.data(name="a", shape=[None, 1], dtype='int64')
b = fluid.data(name="b", shape=[None, 1], dtype='int64')

result = layers.elementwise_add(a, b)

cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
# exe.run(fluid.default_startup_program())

# data_1 = int(input("Please enter an integer: a="))
# data_2 = int(input("Please enter an integer: b="))
data_1 = 1000
data_2 = 300
x = numpy.array([[data_1], [10]], dtype='int64')
y = numpy.array([[data_2], [200]], dtype='int64')

ret = exe.run(
    feed={'a': x, 'b': y},  # 将输入数据x, y分别赋值给变量a，b
    fetch_list=[result]  # 通过fetch_list参数指定需要获取的变量结果
)

print('{}'.format(ret))
