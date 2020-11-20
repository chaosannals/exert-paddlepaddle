import os
import numpy
import paddle.fluid as fluid
from importlib import import_module
parse_args = import_module('args').parse_args

def infer(use_cuda, params_dirname=None):
    from PIL import Image
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    def load_image(infer_file):
        im = Image.open(infer_file)
        im = im.resize((32, 32), Image.ANTIALIAS)

        im = numpy.array(im).astype(numpy.float32)
        # The storage order of the loaded image is W(width),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1))  # CHW
        im = im / 255.0

        # Add one dimension to mimic the list format.
        im = numpy.expand_dims(im, axis=0)
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/../../asset/dog.png')

    with fluid.scope_guard(inference_scope):
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets)

        # infer label
        label_list = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
            "horse", "ship", "truck"
        ]

        print("infer results: %s" % label_list[numpy.argmax(results[0])])

def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "model/imgcls-infer"

    infer(use_cuda=use_cuda, params_dirname=save_path)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu
    main(use_cuda)