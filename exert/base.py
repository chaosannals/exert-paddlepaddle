import os
import paddle.fluid as fluid


class BaseModel:
    '''
    '''


class BaseProgram:
    '''
    '''

    def __init__(self, path, place='cpu'):
        '''
        '''

        self.path = path
        self.place = fluid.CUDAPlace() if place != 'cpu' else fluid.CPUPlace()

    def train(self, feed, fetch_list):
        '''
        训练。
        '''

        executor = fluid.Executor(self.place)
        train_program = fluid.Program()
        startup_program = fluid.Program()

        # 定义模型
        with fluid.program_guard(train_program, startup_program):
            data = fluid.data(name='X', shape=[None, 1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)

        # 启动初始程序
        executor.run(startup_program)

        # 加载模型信息
        if os.path.isdir(self.path):
            fluid.io.load_persistables(
                executor=executor,
                dirname=self.path,
                main_program=train_program,
            )

        # 执行
        result = executor.run(
            train_program,
            feed=feed,
            fetch_list=fetch_list
        )

        # 保存模型
        fluid.io.save_persistables(
            executor=executor,
            dirname=self.path,
            main_program=train_program,
        )
        return result

    def infer(self):
        '''
        '''

        executor = fluid.Executor(self.place)
        infer_program = fluid.Program()
        startup_program = fluid.Program()

        # 启动初始程序
        executor.run(startup_program)

        # 加载模型信息
        fluid.io.load_persistables(
            executor=executor,
            dirname=self.path,
            main_program=infer_program
        )
