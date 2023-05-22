import paddle

src_idx2word = paddle.dataset.wmt16.get_dict(
        "en", 30000, reverse=True)

print(src_idx2word)
