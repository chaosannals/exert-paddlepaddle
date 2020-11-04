import os
import paddle.fluid as fluid
import numpy

# 模型初始化。
train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    data = fluid.data(name='X', shape=[None, 1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    sgd = fluid.optimizer.SGD(learning_rate=0.001)
    sgd.minimize(loss)

# 执行器。
exe = fluid.Executor(fluid.CPUPlace())
# 加载前，执行模型初始化。
exe.run(startup_program)

# 如果模型存在，加载模型。
mdir = "model/single_params"
if os.path.isdir(mdir):
    fluid.io.load_params(
        executor=exe,
        dirname=mdir,
        main_program=train_program
    )

# 生成随机用来测试的数据。
x = numpy.random.random(size=(10, 1)).astype('float32')
# 运行训练程序。
loss_data, = exe.run(train_program,
                     feed={"X": x},
                     fetch_list=[loss.name])

print(f'{loss_data}')

# 保存模型
fluid.io.save_params(
    executor=exe,
    dirname=mdir,
    main_program=train_program
)