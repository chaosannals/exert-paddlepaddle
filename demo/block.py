import paddle.fluid as fluid
import numpy as np
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
    b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
    hidden_w = fluid.layers.matmul(x=data, y=w)
    hidden_b = fluid.layers.elementwise_add(hidden_w, b)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# 读出参数、变量，后设置。
for block in main_prog.blocks:
    for param in block.all_parameters():
        pd_var = fluid.global_scope().find_var(param.name)
        pd_param = pd_var.get_tensor()
        print(f"load: {param.name}, shape: {param.shape}")
        print("设置前的值（前 5 个）: {}".format(np.array(pd_param).ravel()[:5]))
        pd_param.set(np.ones(param.shape), place)
        print("设置后的值（前 5 个）: {}".format(np.array(pd_param).ravel()[:5]))