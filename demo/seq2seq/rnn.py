import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

dict_size = 30000
source_dict_size = target_dict_size = dict_size
bos_id = 0
eos_id = 1
word_dim = 512
hidden_dim = 512
decoder_size = hidden_dim
max_length = 256
beam_size = 4
batch_size = 64

model_save_dir = "model/seq2seq-infer"

class DecoderCell(layers.RNNCell):
    """Additive Attention followed by GRU"""

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.gru_cell = layers.GRUCell(hidden_size)

    def attention(self, hidden, encoder_output, encoder_output_proj,
                  encoder_padding_mask):
        decoder_state_proj = layers.unsqueeze(
            layers.fc(hidden, size=self.hidden_size, bias_attr=False), [1])
        mixed_state = fluid.layers.elementwise_add(
            encoder_output_proj,
            layers.expand(decoder_state_proj,
                          [1, layers.shape(decoder_state_proj)[1], 1]))
        # attn_scores: [batch_size, src_seq_len]
        attn_scores = layers.squeeze(
            layers.fc(
                input=mixed_state, size=1, num_flatten_dims=2, bias_attr=False),
            [2])
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores,
                                                 encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        context = layers.reduce_sum(
            layers.elementwise_mul(encoder_output, attn_scores, axis=0), dim=1)
        return context

    def call(self,
             step_input,
             hidden,
             encoder_output,
             encoder_output_proj,
             encoder_padding_mask=None):
        context = self.attention(hidden, encoder_output, encoder_output_proj,
                                 encoder_padding_mask)
        step_input = layers.concat([step_input, context], axis=1)
        output, new_hidden = self.gru_cell(step_input, hidden)
        return output, new_hidden



def data_func(is_train=True):
    """data inputs and data loader"""
    src = fluid.data(name="src", shape=[None, None], dtype="int64")
    src_sequence_length = fluid.data(
        name="src_sequence_length", shape=[None], dtype="int64")
    inputs = [src, src_sequence_length]
    if is_train:
        trg = fluid.data(name="trg", shape=[None, None], dtype="int64")
        trg_sequence_length = fluid.data(
            name="trg_sequence_length", shape=[None], dtype="int64")
        label = fluid.data(name="label", shape=[None, None], dtype="int64")
        inputs += [trg, trg_sequence_length, label]
    loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=10, iterable=True, use_double_buffer=True)
    return inputs, loader


def encoder(src_embedding, src_sequence_length):
    """Encoder: Bidirectional GRU"""
    encoder_fwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    encoder_fwd_output, fwd_state = layers.rnn(
        cell=encoder_fwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,
        is_reverse=False)
    encoder_bwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    encoder_bwd_output, bwd_state = layers.rnn(
        cell=encoder_bwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,
        is_reverse=True)
    encoder_output = layers.concat(
        input=[encoder_fwd_output, encoder_bwd_output], axis=2)
    encoder_state = layers.concat(input=[fwd_state, bwd_state], axis=1)
    return encoder_output, encoder_state


def decoder(encoder_output,
            encoder_output_proj,
            encoder_state,
            encoder_padding_mask,
            trg=None,
            is_train=True):
    """Decoder: GRU with Attention"""
    decoder_cell = DecoderCell(hidden_size=decoder_size)
    decoder_initial_states = layers.fc(
        encoder_state, size=decoder_size, act="tanh")
    trg_embeder = lambda x: fluid.embedding(input=x,
                                            size=[target_dict_size, hidden_dim],
                                            dtype="float32",
                                            param_attr=fluid.ParamAttr(
                                                name="trg_emb_table"))
    output_layer = lambda x: layers.fc(x,
                                       size=target_dict_size,
                                       num_flatten_dims=len(x.shape) - 1,
                                       param_attr=fluid.ParamAttr(name=
                                                                  "output_w"))
    if is_train:
        decoder_output, _ = layers.rnn(
            cell=decoder_cell,
            inputs=trg_embeder(trg),
            initial_states=decoder_initial_states,
            time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask)
        decoder_output = output_layer(decoder_output)
    else:
        encoder_output = layers.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, beam_size)
        encoder_output_proj = layers.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output_proj, beam_size)
        encoder_padding_mask = layers.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, beam_size)
        beam_search_decoder = layers.BeamSearchDecoder(
            cell=decoder_cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=trg_embeder,
            output_fn=output_layer)
        decoder_output, _ = layers.dynamic_decode(
            decoder=beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=max_length,
            output_time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask)

    return decoder_output


def model_func(inputs, is_train=True):
    src = inputs[0]
    src_sequence_length = inputs[1]
    # source embedding
    src_embeder = lambda x: fluid.embedding(
        input=x,
        size=[source_dict_size, hidden_dim],
        dtype="float32",
        param_attr=fluid.ParamAttr(name="src_emb_table"))
    src_embedding = src_embeder(src)

    # encoder
    encoder_output, encoder_state = encoder(src_embedding, src_sequence_length)

    encoder_output_proj = layers.fc(
        input=encoder_output,
        size=decoder_size,
        num_flatten_dims=2,
        bias_attr=False)
    src_mask = layers.sequence_mask(
        src_sequence_length, maxlen=layers.shape(src)[1], dtype="float32")
    encoder_padding_mask = (src_mask - 1.0) * 1e9

    trg = inputs[2] if is_train else None

    # decoder
    output = decoder(
        encoder_output=encoder_output,
        encoder_output_proj=encoder_output_proj,
        encoder_state=encoder_state,
        encoder_padding_mask=encoder_padding_mask,
        trg=trg,
        is_train=is_train)
    return output


def loss_func(logits, label, trg_sequence_length):
    probs = layers.softmax(logits)
    loss = layers.cross_entropy(input=probs, label=label)
    trg_mask = layers.sequence_mask(
        trg_sequence_length, maxlen=layers.shape(logits)[1], dtype="float32")
    avg_cost = layers.reduce_sum(loss * trg_mask) / layers.reduce_sum(trg_mask)
    return avg_cost


def optimizer_func():
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(
        clip_norm=5.0))
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_dim, 1000)
    return fluid.optimizer.Adam(
        learning_rate=lr_decay,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-4))


def inputs_generator(batch_size, pad_id, is_train=True):
    data_generator = fluid.io.shuffle(
        paddle.dataset.wmt16.train(source_dict_size, target_dict_size),
        buf_size=10000) if is_train else paddle.dataset.wmt16.test(
            source_dict_size, target_dict_size)
    batch_generator = fluid.io.batch(data_generator, batch_size=batch_size)

    def _pad_batch_data(insts, pad_id):
        seq_lengths = np.array(list(map(len, insts)), dtype="int64")
        max_len = max(seq_lengths)
        pad_data = np.array(
            [inst + [pad_id] * (max_len - len(inst)) for inst in insts],
            dtype="int64")
        return pad_data, seq_lengths

    def _generator():
        for batch in batch_generator():
            batch_src = [ins[0] for ins in batch]
            src_data, src_lengths = _pad_batch_data(batch_src, pad_id)
            inputs = [src_data, src_lengths]
            if is_train:
                batch_trg = [ins[1] for ins in batch]
                trg_data, trg_lengths = _pad_batch_data(batch_trg, pad_id)
                batch_lbl = [ins[2] for ins in batch]
                lbl_data, _ = _pad_batch_data(batch_lbl, pad_id)
                inputs += [trg_data, trg_lengths, lbl_data]
            yield inputs

    return _generator
