from util import Record, partial
import tensorflow as tf


scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)

def attention(query, value, mask, dim, head=8):
    """computes scaled dot-product attention
    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, dim, t)
    `dim` must be divisible by `head`
    """
    assert not dim % head
    d,h,c = dim, head, dim // head
    b,_,t = get_shape(query)
    b,_,s = get_shape(value)
    # pretransformations
    v = tf.reshape(tf.layers.dense(value, dim, name='v', use_bias=False), (b,h,c,s)) # bhcs <- bds <- bvs
    k = tf.reshape(tf.layers.dense(value, dim, name='k', use_bias=False), (b,h,c,s)) # bhcs <- bds <- bvs
    q = tf.reshape(tf.layers.dense(query, dim, name='q', use_bias=False), (b,h,c,s)) # bhct <- bdt <- bqt
    # weight
    a = tf.matmul(q, k, transpose_a= True) # bhts <- (bhtc <- bhct) @ bhcs
    a *= c ** -0.5
    if mask is not None: a += tf.expand_dims(mask, axis= 1)
    a = tf.nn.softmax(a, axis= -1)
    # attend
    y = tf.matmul(v, a, transpose_b= True) # bhct <- bhcs @ (bhst <- bhts)
    # posttransformation
    return tf.layers.dense(tf.reshape(y, (b,d,t)), dim, name='p', use_bias=False) # bdt <- bdt <- bhct

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
