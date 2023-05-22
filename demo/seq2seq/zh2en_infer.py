import paddle
import paddle.fluid as fluid

def data_func():
    """data inputs and data loader"""
    src = fluid.data(name="src", shape=[None, None], dtype="int64")
    src_sequence_length = fluid.data(
        name="src_sequence_length", shape=[None], dtype="int64")
    inputs = [src, src_sequence_length]
    loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=10, iterable=True, use_double_buffer=True)
    return inputs, loader

# inputs, loader = data_func()

# print(inputs)
# print(loader)

i = paddle.dataset.wmt16.train(300, 300)
j = fluid.io.batch(i, 1)
print([k for k in j()])