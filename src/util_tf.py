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
