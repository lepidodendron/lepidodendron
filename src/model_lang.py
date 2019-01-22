from util import Record
from util_tf import tf, scope, placeholder


def model(mode
          , width
          , height
          , tgt= None
          , len_tgt= None
          , num_layers= 3
          , num_units= 512
          , learn_rate= 1e-3
          , decay_rate= 1e-2
          , dropout= 0.1):
    assert mode in ('train', 'valid', 'infer')
    self = Record()

    with scope('target'):
        tgt     = self.tgt     = placeholder(tf.uint8, (None, height, None, width), tgt, 'tgt') # n h t w
        len_tgt = self.len_tgt = placeholder(tf.int32, (None,), len_tgt, 'len_tgt') # n
        max_tgt = self.max_tgt = tf.reduce_max(len_tgt)
        num_tgt = tf.size(len_tgt)
        # tgt = tgt[:,:,:max_tgt,:]
        tgt = tf.transpose(tgt, (2, 0, 1, 3))                     # t n h w
        tgt = tf.reshape(tgt, (max_tgt, num_tgt, height * width)) # t n hw
        tgt = tf.to_float(tgt) / 255.0
        true = self.true = tf.pad(tgt, ((0,1),(0,0),(0,0)))
        fire = self.fire = tf.pad(tgt, ((1,0),(0,0),(0,0)))
        mask_tgt = self.mask_tgt = tf.sequence_mask(1+len_tgt, 1+max_tgt)

    with scope('decode'):
        decoder  = self.decoder  = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units, dropout= dropout)
        state_in = self.state_in = tf.zeros((num_layers, num_tgt, num_units))
        y, _ = _, (self.state_ex,) = decoder(fire, initial_state= (state_in,), training= 'train' == mode)

    # todo predict when to stop based on y and mask_tgt

    if 'infer' != mode:
        y = tf.boolean_mask(y, mask_tgt)
        true = tf.boolean_mask(true, mask_tgt)

    with scope('output'):
        y = tf.layers.dense(y, height * width, name= 'logits')
        pred = self.pred = tf.sigmoid(y)

    # todo character recognition network

    with scope('losses'):
        diff = true - pred
        sae  = self.sae  = tf.reduce_sum(tf.abs(diff), axis= -1)
        sse  = self.sse  = tf.reduce_sum(tf.square(diff), axis= -1)
        xent = self.xent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits= y, labels= true), axis= -1)
        loss = tf.reduce_mean(xent) # todo try other losses

    with scope('update'):
        step = self.step = tf.train.get_or_create_global_step()
        lr = self.lr = learn_rate / (1.0 + decay_rate * tf.sqrt(tf.to_float(step)))
        if 'train' == mode:
            down = self.down = tf.train.AdamOptimizer(lr).minimize(loss, step)

    return self


if '__main__' == __name__:

    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    from tqdm import tqdm
    from util_cw import CharWright
    from util_io import load_txt
    from util_np import np, batch_sample, partition
    from util_tf import pipe

    tf.set_random_seed(0)
    sess = tf.InteractiveSession()

    cw = CharWright.load("../data/cw_tgt.pkl")
    tgt = np.array(list(load_txt("../data/tgt.txt")))
    valid = tgt[:4096]
    train = tgt[4096:]

    def batch(size= 256, seed= 0):
        for bat in batch_sample(len(train), size, seed):
            yield cw.write(train[bat])

    tgt, len_tgt = pipe(batch, (tf.uint8, tf.int32), prefetch= 16)
    model_train = model('train', width= cw.width, height= cw.height, tgt= tgt, len_tgt= len_tgt)
    model_valid = model('valid', width= cw.width, height= cw.height)
    dummy = tuple(placeholder(tf.float32, ()) for _ in range(3))

    def log(step
            , wtr= tf.summary.FileWriter("/cache/tensorboard-logdir/lepidodendron/l")
            , log= tf.summary.merge(
                (  tf.summary.scalar('step_sae' , dummy[0])
                 , tf.summary.scalar('step_sse' , dummy[1])
                 , tf.summary.scalar('step_xent', dummy[2])))
            , input= (model_valid.tgt, model_valid.len_tgt)
            , fetch= (model_valid.sae, model_valid.sse, model_valid.xent)
            , batch= 512):
        stats = [sess.run(fetch, dict(zip(input, cw.write(valid[i:j])))) for i, j in partition(len(valid), batch)]
        stats = [np.mean(np.concatenate(stat)) for stat in zip(*stats)]
        wtr.add_summary(sess.run(log, dict(zip(dummy, stats))), step)
        wtr.flush()

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    # saver.restore(sess, "../ckpt/l9")

    for r in range(10):
        for _ in range(100): # 10k steps per round
            for _ in tqdm(range(100), ncols= 70):
                sess.run(model_train.down)
            log(sess.run(model_train.step))
        saver.save(sess, "../ckpt/l{}".format(r), write_meta_graph= False)
