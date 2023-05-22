import six
import paddle.fluid as fluid
import importlib
rnn = importlib.import_module('rnn')
data_func = rnn.data_func
model_func = rnn.model_func
loss_func = rnn.loss_func
optimizer_func = rnn.optimizer_func
inputs_generator = rnn.inputs_generator
batch_size = rnn.batch_size
eos_id = rnn.eos_id
model_save_dir = rnn.model_save_dir


def train(use_cuda):
    # define program
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # For training:
            # inputs = [src, src_sequence_length, trg, trg_sequence_length, label]
            inputs, loader = data_func(is_train=True)
            logits = model_func(inputs, is_train=True)
            loss = loss_func(logits, inputs[-1], inputs[-2])
            optimizer = optimizer_func()
            optimizer.minimize(loss)

    # define data source
    places = fluid.cuda_places() if use_cuda else fluid.cpu_places()
    loader.set_batch_generator(
        inputs_generator(batch_size, eos_id, is_train=True), places=places)

    exe = fluid.Executor(places[0])
    exe.run(startup_prog)
    prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name)

    EPOCH_NUM = 20
    for pass_id in six.moves.xrange(EPOCH_NUM):
        batch_id = 0
        for data in loader():
            loss_val = exe.run(prog, feed=data, fetch_list=[loss])[0]
            print('pass_id: %d, batch_id: %d, loss: %f' %
                  (pass_id, batch_id, loss_val))
            batch_id += 1
        fluid.io.save_params(exe, model_save_dir, main_program=train_prog)


def main(use_cuda=False):
    train(use_cuda)


if __name__ == '__main__':
    main()
