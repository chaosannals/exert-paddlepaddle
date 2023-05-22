import numpy as np
import paddle
import paddle.fluid as fluid
import importlib
rnn = importlib.import_module('rnn')
model_func = rnn.model_func
inputs_generator = rnn.inputs_generator
batch_size = rnn.batch_size
eos_id = rnn.eos_id
source_dict_size = rnn.source_dict_size
target_dict_size = rnn.target_dict_size
model_save_dir = rnn.model_save_dir
beam_size = rnn.beam_size


def data_func():
    """data inputs and data loader"""
    src = fluid.data(name="src", shape=[None, None], dtype="int64")
    src_sequence_length = fluid.data(
        name="src_sequence_length", shape=[None], dtype="int64")
    inputs = [src, src_sequence_length]
    loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=10, iterable=True, use_double_buffer=True)
    return inputs, loader

def infer(use_cuda):
    # define program
    infer_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs, loader = data_func()
            predict_seqs = model_func(inputs, is_train=False)

    # define data source
    places = fluid.cuda_places() if use_cuda else fluid.cpu_places()
    loader.set_batch_generator(
        inputs_generator(batch_size, eos_id, is_train=False), places=places)
    src_idx2word = paddle.dataset.wmt16.get_dict(
        "en", source_dict_size, reverse=True)
    trg_idx2word = paddle.dataset.wmt16.get_dict(
        "de", target_dict_size, reverse=True)

    exe = fluid.Executor(places[0])
    exe.run(startup_prog)
    fluid.io.load_params(exe, model_save_dir, main_program=infer_prog)
    prog = fluid.CompiledProgram(infer_prog).with_data_parallel()

    for data in loader():
        seq_ids = exe.run(prog, feed=data, fetch_list=[predict_seqs])[0]
        for ins_idx in range(seq_ids.shape[0]):
            print("Original sentence:")
            src_seqs = np.array(data[0]["src"])
            print(" ".join([
                src_idx2word[idx] for idx in src_seqs[ins_idx][1:]
                if idx != eos_id
            ]))
            print("Translated sentence:")
            for beam_idx in range(beam_size):
                seq = [
                    trg_idx2word[idx] for idx in seq_ids[ins_idx, :, beam_idx]
                    if idx != eos_id
                ]
                print(" ".join(seq).encode("utf8"))


def main(use_cuda):
    infer(use_cuda)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)
