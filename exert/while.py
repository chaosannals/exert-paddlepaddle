from paddle import fluid
from paddle.fluid import layers
import numpy

def cond(i, ten):
    return i < ten

def body(i, dummy):
    i = i + 2
    return i, dummy

i = layers.fill_constant(shape=[1], value=0, dtype='int64')
ten = layers.fill_constant(shape=[1], value=10, dtype='int64')
out, ten = layers.while_loop(cond=cond, body=body, loop_vars=[i, ten])

