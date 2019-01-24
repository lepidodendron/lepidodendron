from util import Record
from util_tf import tf, scope, placeholder, flatten


def model(mode
          , nchars
          , width
          , height
          , tgt_img= None
          , tgt_idx= None
          , len_tgt= None
          , num_layers= 3
          , num_units= 512
          , learn_rate= 1e-3
          , decay_rate= 1e-2
          , dropout= 0.1):
    assert mode in ('train', 'valid', 'infer')
    self = Record()

    with scope('target'):
        # input nodes
        tgt_img = self.tgt_img = placeholder(tf.uint8, (None, height, None, width), tgt_img, 'tgt_img') # n h t w
        tgt_idx = self.tgt_idx = placeholder(tf.int32, (None,         None       ), tgt_idx, 'tgt_idx') # n t
        len_tgt = self.len_tgt = placeholder(tf.int32, (None,                    ), len_tgt, 'len_tgt') # n

        # time major order
        tgt_idx = tf.transpose(tgt_idx) # t n
        tgt_img = tf.transpose(tgt_img, (2, 0, 1, 3)) # t n h w
        tgt_img = flatten(tgt_img, 2, 3) # t n hw

        # normalize pixels to binary
        tgt_img = tf.to_float(tgt_img) / 255.0
        # tgt_img = tf.round(tgt_img)
        # todo consider adding noise

        # causal padding
        fire = self.fire = tf.pad(tgt_img, ((1,0),(0,0),(0,0)), constant_values= 0.0)
        true = self.true = tf.pad(tgt_img, ((0,1),(0,0),(0,0)), constant_values= 1.0)
        tidx = self.tidx = tf.pad(tgt_idx, ((0,1),(0,0))      , constant_values= 0  )
        mask_tgt = tf.transpose(tf.sequence_mask(len_tgt + 1)) # t n

    with scope('decode'):
        decoder  = self.decoder  = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units, dropout= dropout)
        state_in = self.state_in = tf.zeros((num_layers, tf.shape(fire)[1], num_units))
        x, _ = _, (self.state_ex,) = decoder(fire, initial_state= (state_in,), training= 'train' == mode)

    if 'infer' != mode:
        x    = tf.boolean_mask(x   , mask_tgt)
        true = tf.boolean_mask(true, mask_tgt)
        tidx = tf.boolean_mask(tidx, mask_tgt)

    with scope('output'):
        y = tf.layers.dense(x, height * width, name= 'logit_img')
        z = tf.layers.dense(x, nchars        , name= 'logit_idx')
        pred = self.pred = tf.sigmoid(y) # todo try regression
        prob = self.prob = tf.nn.softmax(z)
        pidx = self.pidx = tf.argmax(z, axis= -1, output_type= tf.int32)

    with scope('losses'):
        diff = true - pred
        mae = self.mae = tf.reduce_mean(tf.abs(diff), axis= -1)
        mse = self.mse = tf.reduce_mean(tf.square(diff), axis= -1)
        xmg = self.xmg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= y, labels= true), axis= -1)
        xid = self.xid = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels= tidx)
        err = self.err = tf.not_equal(tidx, pidx)
        loss = tf.reduce_mean(xid) + tf.reduce_mean(xmg)

    with scope('update'):
        step = self.step = tf.train.get_or_create_global_step()
        lr = self.lr = learn_rate / (1.0 + decay_rate * tf.sqrt(tf.to_float(step)))
        if 'train' == mode:
            down = self.down = tf.train.AdamOptimizer(lr).minimize(loss, step)

    return self


if '__main__' == __name__:

    trial = 'lang'
    ckpt  =  None

    from tqdm import tqdm
    from util_cw import CharWright
    from util_io import load_txt
    from util_np import np, batch_sample, partition
    from util_tf import pipe

    tf.set_random_seed(0)
    sess = tf.InteractiveSession()

    cwt = CharWright.load("../data/cwt.pkl")
    tgt = np.array(list(load_txt("../data/tgt.txt")))
    tgt_valid = tgt[:4096]
    tgt_train = tgt[4096:]

    def batch(tgt= tgt_train, cwt= cwt, size= 256, seed= 0):
        for bat in batch_sample(len(tgt), size, seed):
            yield cwt(tgt[bat])

    tgt_img, tgt_idx, len_tgt = pipe(batch, (tf.uint8, tf.int32, tf.int32))
    train = model('train', cwt.nchars(), cwt.width, cwt.height, tgt_img, tgt_idx, len_tgt)
    valid = model('valid', cwt.nchars(), cwt.width, cwt.height)
    dummy = tuple(placeholder(tf.float32, ()) for _ in range(5))

    def log(step
            , wtr= tf.summary.FileWriter("/cache/tensorboard-logdir/lepidodendron/{}".format(trial))
            , log= tf.summary.merge(
                (  tf.summary.scalar('step_mae', dummy[0])
                 , tf.summary.scalar('step_mse', dummy[1])
                 , tf.summary.scalar('step_xmg', dummy[2])
                 , tf.summary.scalar('step_xid', dummy[3])
                 , tf.summary.scalar('step_err', dummy[4])))
            , fet= (valid.mae, valid.mse, valid.xmg, valid.xid, valid.err)
            , inp= (valid.tgt_img, valid.tgt_idx, valid.len_tgt)
            , tgt= tgt_valid
            , cwt= cwt
            , bat= 512):
        stats = [sess.run(fet, dict(zip(inp, cwt(tgt[i:j])))) for i, j in partition(len(tgt), bat)]
        stats = [np.mean(np.concatenate(stat)) for stat in zip(*stats)]
        wtr.add_summary(sess.run(log, dict(zip(dummy, stats))), step)
        wtr.flush()

    saver = tf.train.Saver()
    if ckpt is not None:
        saver.restore(sess, "../ckpt/{}_{}".format(trial, ckpt))
    else:
        tf.global_variables_initializer().run()

    for ckpt in range(37):
        for _ in range(40): # 10k steps per round
            for _ in tqdm(range(250), ncols= 70):
                sess.run(train.down)
            log(sess.run(train.step))
        saver.save(sess, "../ckpt/{}_{}".format(trial, ckpt), write_meta_graph= False)
