# from chainer import cuda
import chainer
import numpy
# import pytest
import chainer.computational_graph as c
# from chainer import function_hooks
from chainer import gradient_check

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.gat import GraphAttentionNetworks
from chainer_chemistry.models.nfp import NFP
from chainer_chemistry.models.rsgcn import RSGCN

atom_size = 5
out_dim = 4
batch_size = 2


model = GraphAttentionNetworks(out_dim=out_dim)
# model = NFP(out_dim=out_dim)
# model = RSGCN(out_dim=out_dim)


atom_data = numpy.random.randint(
    0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
).astype(numpy.int32)
adj_data = numpy.random.randint(
    0, high=2, size=(batch_size, atom_size, atom_size)
).astype(numpy.float32)
y_grad = numpy.random.uniform(
    -1, 1, (batch_size, out_dim)).astype(numpy.float32)


gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                              params=tuple(model.params()),
                              rtol=0.001, no_grads=[True, True])

# atom_data = chainer.Variable(atom_data)
# adj_data = chainer.Variable(adj_data)
#
# with chainer.function_hooks.PrintHook():
#     a = model(atom_data, adj_data)
#     a.grad = y_grad
#     a.backward()
# g = c.build_computational_graph(a)
# with open('nfp.dot', 'w') as o:
#     o.write(g.dump())

# a = model(atom_data, adj_data)
# a.grad = y_grad
# a.backward()
