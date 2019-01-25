from util import Record, partial
import tensorflow as tf


scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(gen_func, gen_types, map_func= None, map_types= None, para_map= 4, prefetch= 4, name= 'pipe'):
    """returns iterator tensors of `gen_types` from generator `gen_func`.
    see `tf.data.Dataset.from_generator`.

    when specified, `map_func` is called on the generator outputs (as
    numpy arrays) and tensors of `map_types` are returned instead.
    `para_map` number of calls are processed in parallel.  `map_func`
    must be stateless.  otherwise simply transform the data in
    `gen_func`.  it should be used only for parallelizing heavy
    transformations.  see `tf.data.Dataset.map` and `tf.py_func`.

    """
    with scope(name):
        ds = tf.data.Dataset.from_generator(gen_func, gen_types)
        if map_func is not None:
            ds = ds.map(
                lambda *args: tf.py_func(map_func, args, map_types, stateful= False)
                , num_parallel_calls= para_map)
        return ds.prefetch(prefetch).make_one_shot_iterator().get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`

    if tensor `x` is given, converts and uses it as default

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def get_shape(x, name= 'shape'):
    """returns the shape of `x` as a tuple of integers (static) or int32
    scalar tensors (dynamic)

    """
    with scope(name):
        shape = tf.shape(x)
        shape = tuple(d if d is not None else shape[i] for i, d in enumerate(x.shape.as_list()))
        return shape


def flatten(x, *axes, name= 'flatten'):
    with scope(name):
        dims = get_shape(x)
        z = len(dims)
        for a in axes:
            if a < -z or z <= a:
                raise ValueError("axis {} out of rank {}".format(a, z))
        axes = {(a + z) % z for a in axes}
        if not axes: axes = set(range(z))
        shape, n = [], None
        for a, d in enumerate(dims):
            if a in axes:
                if n is None:
                    n = d
                else:
                    n *= d
            else:
                if n is not None:
                    shape.append(n)
                    n = None
                shape.append(d)
        if n is not None:
            shape.append(n)
        return tf.reshape(x, shape)


def variable(name, shape, init= 'rand', initializers=
             {  'zero': tf.initializers.zeros()
              , 'unit': tf.initializers.ones()
              , 'rand': tf.glorot_uniform_initializer()
             }):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


class Conv(Record):
    """convolution from `m` to `n` channels

    the default parameters make a position-wise linear layer

    """

    def __init__(self, n, m= None, size= 1, name= 'conv'):
        if m is None: m = n
        self.name = name
        with scope(name):
            self.kern = variable('kern', (size, m, n))

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.nn.convolution(x, self.kern, padding= 'VALID', data_format= 'NCW')

    def shape(self):
        return get_shape(self.kern)


class Attention(Record):
    """computes multi-head scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, d_q, t)

    `dim` must be divisible by `head`

    `mask` has on-values 0 and off-values -inf

    """

    def __init__(self, dim, d_q= None, d_v= None, head= 8, name= 'attention'):
        assert not dim % head
        if d_q is None: d_q = dim
        if d_v is None: d_v = dim
        self.dim = dim
        self.head = head
        self.name = name
        with scope(name):
            self.v = Conv(dim, d_v, name= 'v')
            self.k = Conv(dim, d_v, name= 'k')
            self.q = Conv(dim, d_q, name= 'q')
            self.p = Conv(d_q, dim, name= 'p')

    def __call__(self, query, value, mask= None, name= None):
        with scope(name or self.name):
            d,h,c = self.dim, self.head, self.dim // self.head
            b,_,t = get_shape(query)
            b,_,s = get_shape(value)
            # pretransformations
            v = tf.reshape(self.v(value), (b,h,c,s)) # bhcs <- bds <- bvs
            k = tf.reshape(self.k(value), (b,h,c,s)) # bhcs <- bds <- bvs
            q = tf.reshape(self.q(query), (b,h,c,t)) # bhct <- bdt <- bqt
            # weight
            a = tf.matmul(q, k, transpose_a= True) # bhts <- (bhtc <- bhct) @ bhcs
            a *= c ** -0.5
            if mask is not None: a += tf.expand_dims(mask, axis= 1)
            a = tf.nn.softmax(a, axis= -1)
            # attend
            y = tf.matmul(v, a, transpose_b= True) # bhct <- bhcs @ (bhst <- bhts)
            # posttransformation
            return self.p(tf.reshape(y, (b,d,t))) # bqt <- bdt <- bhct
