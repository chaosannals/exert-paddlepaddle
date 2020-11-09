import sys
import math
import paddle
import paddle.fluid as fluid
from importlib import import_module

# 训练测试
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]

def trans(args, place, params_dirname):
    '''
    训练。
    '''

    batch_size = 20

    # 加载数据。
    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(),
            batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(),
            batch_size=batch_size
        )
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.train(),
                buf_size=500
            ),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.test(), buf_size=500),
            batch_size=batch_size)

    # feature vector of length 13
    x = fluid.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='float32')

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        startup_program.random_seed = 90

    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    test_program = main_program.clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    exe = fluid.Executor(place)
    num_epochs = args.num_epochs

    # 主训练。
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    step = 0

    exe_test = fluid.Executor(place)

    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            if step % 10 == 0:  # record a train cost every 10 batches
                print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))

            if step % 100 == 0:  # record a test cost every 100 batches
                test_metics = train_test(
                    executor=exe_test,
                    program=test_program,
                    reader=test_reader,
                    fetch_list=[avg_loss],
                    feeder=feeder)
                print("%s, Step %d, Cost %f" %
                      (test_prompt, step, test_metics[0]))
                # If the accuracy is good enough, we can stop the training.
                if test_metics[0] < 10.0:
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        if params_dirname is not None:
            # We can save the trained parameters for the inferences later
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)

        if args.enable_ce and pass_id == args.num_epochs - 1:
            print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
            print("kpis\ttest_cost\t%f" % test_metics[0])

def main(args):
    params_dirname = "model/fitaline"
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    # 训练
    trans(args, place, params_dirname)

if __name__ == '__main__':
    args = import_module('args')
    main(args.parse_args())