import numpy
from chainer import functions
# import chainer_chemistry
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear

atom_size = 5
out_dim = 4
batch_size = 3
heads = 2
hidden_dim = 16

atom_data = numpy.random.randint(
    0, high=110, size=(batch_size, atom_size)
).astype(numpy.int32)
adj_data = numpy.random.randint(
    0, high=2, size=(batch_size, atom_size, atom_size)
).astype(numpy.float32)


embed = EmbedAtomID(out_size=hidden_dim, in_size=110)
weight = GraphLinear(hidden_dim, heads * hidden_dim)
att_weight = GraphLinear(hidden_dim * 2, 1)


def test(atom_array, adj_data):
    x = embed(atom_array)
    mb, atom, ch = x.shape
    print(x.shape)
    test = weight(x)
    print(test.shape)
    x = functions.expand_dims(test, axis=1)
    print(x.shape)
    x = functions.broadcast_to(x, (mb, atom, atom, heads * ch))
    print(x.shape)
    y = functions.copy(x, -1)
    print(y.shape)
    y = functions.transpose(y, (0, 2, 1, 3))
    print(y.shape)
    z = functions.concat([x, y], axis=3)
    print(z.shape)
    z = functions.reshape(z, (mb * heads, atom * atom, ch * 2))
    print(z.shape)
    z = att_weight(z)
    print(z.shape)
    z = functions.reshape(z, (mb * heads, atom, atom))
    print(z.shape)
    z = functions.leaky_relu(z)
    # print(adj_data.array.astype(numpy.bool))
    # cond = adj_data.array.astype(numpy.bool)
    cond = adj_data.astype(numpy.bool)
    print(cond.shape)
    z = functions.reshape(z, (heads, mb, atom, atom))
    cond = numpy.broadcast_to(cond, z.array.shape)
    print(cond.shape)
    z = functions.where(cond, z,
                        numpy.broadcast_to(numpy.array(-100), z.array.shape)
                        .astype(numpy.float32))
    print(z.shape)
    z = functions.softmax(z)
    print(z.shape)
    z = functions.mean(z, axis=0)
    print(z.shape)
    print(test.shape)
    z = functions.matmul(z, test)
    return z


a = test(atom_data, adj_data)
print(a)
