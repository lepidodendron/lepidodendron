from util import Record
from util_tf import tf, scope, placeholder, flatten, Attention, Normalize


def model(mode
          , src_dwh
          , tgt_dwh
          # , src_img= None
          , src_idx= None
          , len_src= None
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

    src_d, src_w, src_h = src_dwh
    tgt_d, tgt_w, tgt_h = tgt_dwh
    # print(src_idx.eval())

    with scope('source'):
        # input nodes
        # src_img = self.src_img = placeholder(tf.uint8, (None, None, src_h, src_w), src_img, 'src_img') # n s h w
        src_idx = self.src_idx = placeholder(tf.int32, (None, None), src_idx, 'src_idx') # n s
        len_src = self.len_src = placeholder(tf.int32, (None,                   ), len_src, 'len_src') # n

        src_idx = tf.one_hot(src_idx, src_d, axis=-1) # n s v

        # time major order
        # emb_src = tf.transpose(src_img, (1, 0, 2, 3)) # s n h w
        emb_src = tf.transpose(src_idx, (1, 0 , 2))  # s n v
        # emb_src = flatten(emb_src, 2, 3) # s n hw
        # emb_src = tf.to_float(emb_src) / 255.0

        for i in range(num_layers):
            with scope("rnn{}".format(i + 1)):
                emb_fwd, _ = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, dropout=dropout, name='fwd')(emb_src,
                                                                                                      training='train' == mode)
                emb_bwd, _ = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, dropout=dropout, name='bwd')(tf.reverse_sequence(emb_src,
                                                                                                         len_src,
                                                                                                         seq_axis= 0,
                                                                                                         batch_axis= 1),
                                                                                     training='train' == mode)
            emb_src = tf.concat((emb_fwd, tf.reverse_sequence(emb_bwd, len_src, seq_axis= 0, batch_axis= 1)), axis=-1)
        # emb_src = tf.layers.dense(emb_src, num_units, name= 'reduce_concat') # s n d
        # emb_src = self.emb_src = tf.transpose(emb_src, (1, 2, 0)) # n d s
        emb_src = self.emb_src = tf.transpose(emb_src, (1, 2, 0)) # n v s

    with scope('target'):
        # input nodes
        # tgt_img = self.tgt_img = placeholder(tf.uint8, (None, None, tgt_h, tgt_w), tgt_img, 'tgt_img') # n t h w
        tgt_idx = self.tgt_idx = placeholder(tf.int32, (None, None              ), tgt_idx, 'tgt_idx') # n t
        len_tgt = self.len_tgt = placeholder(tf.int32, (None,                   ), len_tgt, 'len_tgt') # n

        # time major order
        tgt_idx = tf.transpose(tgt_idx) # t n
        # tgt_img = tf.transpose(tgt_img, (1, 0, 2, 3)) # t n h w
        # tgt_img = flatten(tgt_img, 2, 3) # t n hw

        # normalize pixels to binary
        # tgt_img = tf.to_float(tgt_img) / 255.0
        # tgt_img = tf.round(tgt_img)
        # todo consider adding noise

        # causal padding
        # fire = self.fire = tf.pad(tgt_img, ((1,0),(0,0),(0,0)), constant_values= 0.0)
        fire = self.fire = tf.pad(tgt_idx, ((1,0),(0,0)), constant_values= 0.0)
        # true = self.true = tf.pad(tgt_img, ((0,1),(0,0),(0,0)), constant_values= 1.0)
        tidx = self.tidx = tf.pad(tgt_idx, ((0,1),(0,0))      , constant_values= 1  )
        mask_tgt = tf.transpose(tf.sequence_mask(len_tgt + 1)) # t n

    with scope('decode'):
        # needs to get input from latent space to do attention or some shit
        decoder  = self.decoder  = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units, dropout= dropout)
        state_in = self.state_in = tf.zeros((num_layers, tf.shape(fire)[1], num_units))
        x, _ = _, (self.state_ex,) = decoder(fire, initial_state= (state_in,), training= 'train' == mode)
        # transform mask to -inf and 0 in order to simply sum for whatever the fuck happens next
        mask = tf.log(tf.sequence_mask(len_src, dtype= tf.float32)) # n s
        mask = tf.expand_dims(mask, 1) # n 1 s
        # multi-head scaled dot-product attention
        x = tf.transpose(x, (1, 2, 0)) # t n d ---> n d t
        attn = Attention(num_units, num_units, 2*num_units)(x, emb_src, mask)
        if 'train' == mode: attn = tf.nn.dropout(attn, 1 - dropout)
        x = Normalize(num_units)(x + attn)
        x = tf.transpose(x, (2, 0, 1)) # n d t ---> t n d

    if 'infer' != mode:
        x    = tf.boolean_mask(x   , mask_tgt)
        # true = tf.boolean_mask(true, mask_tgt)
        tidx = tf.boolean_mask(tidx, mask_tgt)

    with scope('output'):
        # y = tf.layers.dense(x, tgt_h * tgt_w, name= 'dense_img')
        z = tf.layers.dense(x, tgt_d        , name= 'logit_idx')
        # pred = self.pred = tf.clip_by_value(y, 0.0, 1.0)
        prob = self.prob = tf.nn.softmax(z)
        pidx = self.pidx = tf.argmax(z, axis= -1, output_type= tf.int32)

    with scope('losses'):
        # diff = true - pred
        # mae = self.mae = tf.reduce_mean(tf.abs(diff), axis= -1)
        # mse = self.mse = tf.reduce_mean(tf.square(diff), axis= -1)
        xid = self.xid = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels= tidx)
        err = self.err = tf.not_equal(tidx, pidx)
        # loss = tf.reduce_mean(xid) + tf.reduce_mean(mae) * 10.0
        loss = tf.reduce_mean(xid)

    with scope('update'):
        step = self.step = tf.train.get_or_create_global_step()
        lr = self.lr = learn_rate / (1.0 + decay_rate * tf.sqrt(tf.to_float(step)))
        if 'train' == mode:
            down = self.down = tf.train.AdamOptimizer(lr).minimize(loss, step)

    return self


if '__main__' == __name__:

    trial = 'bale2_c2c'
    ckpt  =  None

    from tqdm import tqdm
    from util_cw import CharWright
    from util_io import load_txt
    from util_np import np, batch_sample, partition
    from util_tf import pipe

    tf.set_random_seed(0)
    sess = tf.InteractiveSession()

    cws = CharWright.load("../data/cws.pkl")
    cwt = CharWright.load("../data/cwt.pkl")
    src = np.array(list(load_txt("../data/src.txt")))
    tgt = np.array(list(load_txt("../data/tgt.txt")))
    src_valid, src_train = src[:4096], src[4096:]
    tgt_valid, tgt_train = tgt[:4096], tgt[4096:]
    val = np.array(sorted(range(len(tgt_valid)), key= lambda i: max(len(src_valid[i]), len(tgt_valid[i]))))
    src_valid = src_valid[val]
    tgt_valid = tgt_valid[val]

    def feed(src, tgt, cws= cws, cwt= cwt):
        # src_img,          len_src = cws(src)
        _      , src_idx, len_src = cws(src, ret_idx= True)
        tgt_img, tgt_idx, len_tgt = cwt(tgt, ret_idx= True)
        # return src_img, len_src, tgt_img, tgt_idx, len_tgt
        return src_idx, len_src, tgt_img, tgt_idx, len_tgt

    def batch(src= src_train, tgt= tgt_train, size= 128, seed= 0):
        for bat in batch_sample(len(tgt), size, seed):
            yield feed(src[bat], tgt[bat])

    # src_img, len_src, tgt_img, tgt_idx, len_tgt = pipe(batch, (tf.uint8, tf.int32, tf.uint8, tf.int32, tf.int32))
    src_idx, len_src, tgt_img, tgt_idx, len_tgt = pipe(batch, (tf.int32, tf.int32, tf.uint8, tf.int32, tf.int32))
    # train = model('train', cws.dwh(), cwt.dwh(), src_img, len_src, tgt_img, tgt_idx, len_tgt)
    train = model('train', cws.dwh(), cwt.dwh(), src_idx, len_src, tgt_img, tgt_idx, len_tgt)
    valid = model('valid', cws.dwh(), cwt.dwh(), src_idx)
    dummy = tuple(placeholder(tf.float32, ()) for _ in range(3))

    def log(step
            , wtr= tf.summary.FileWriter("/cache/tensorboard-logdir/lepidodendron/{}".format(trial))
            , log= tf.summary.merge(
                (  tf.summary.scalar('step_mae', dummy[0])
                 , tf.summary.scalar('step_xid', dummy[1])
                 , tf.summary.scalar('step_err', dummy[2])))
            , fet= (valid.mae, valid.xid, valid.err)
            , inp= (valid.src_idx, valid.len_src, valid.tgt_img, valid.tgt_idx, valid.len_tgt)
            , src= src_valid
            , tgt= tgt_valid
            , bat= 256):
        stats = [sess.run(fet, dict(zip(inp, feed(src[i:j], tgt[i:j])))) for i, j in partition(len(tgt), bat)]
        stats = [np.mean(np.concatenate(stat)) for stat in zip(*stats)]
        wtr.add_summary(sess.run(log, dict(zip(dummy, stats))), step)
        wtr.flush()

    saver = tf.train.Saver()
    if ckpt is not None:
        saver.restore(sess, "../ckpt/{}_{}".format(trial, ckpt))
    else:
        tf.global_variables_initializer().run()

    for ckpt in range(37): # 60 epochs
        for _ in range(40): # 10k steps per round
            for _ in tqdm(range(250), ncols= 70):
                sess.run(train.down)
            log(sess.run(train.step))
        saver.save(sess, "../ckpt/{}_{}".format(trial, ckpt), write_meta_graph= False)
