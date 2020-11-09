import numpy
import paddle
import paddle.fluid as fluid
from importlib import import_module

def save_result(points1, points2):
    '''
    保存结果，生成图。
    '''

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')

def infer(place, params_dirname):
    '''
    预测。
    '''

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # 预测
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10

        infer_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

        infer_data = next(infer_reader())
        infer_feat = numpy.array(
            [data[0] for data in infer_data]).astype("float32")
        infer_label = numpy.array(
            [data[1] for data in infer_data]).astype("float32")

        assert feed_target_names[0] == 'x'
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: numpy.array(infer_feat)},
            fetch_list=fetch_targets)

        print("预测结果: (房价)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        print("\n真实数据:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)

def main(args):
    params_dirname = "model/fitaline"
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    # 预测
    infer(place, params_dirname)

if __name__ == '__main__':
    args = import_module('args')
    main(args.parse_args())