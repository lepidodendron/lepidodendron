from util import Record, identity
from util_np import np, partition
from util_tf import tf, scope, placeholder, flatten, Normalize, Smooth, Dropout, Conv, Attention


def causal_mask(t, name= 'causal_mask'):
    """returns the causal mask for `t` steps"""
    with scope(name):
        return tf.linalg.LinearOperatorLowerTriangular(tf.ones((t, t))).to_dense()


def sinusoid(dim, time, freq= 1e-4, array= False):
    """returns a rank-2 tensor of shape `dim, time`, where each column
    corresponds to a time step and each row a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ (1 + np.arange(time).reshape(1, -1))
        return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
    else:
        assert False # figure out a better way to do this
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                1 + tf.range(tf.to_float(time), dtype= tf.float32)
                , (1, -1))
        return tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), axis= -1), (dim, time))


class Sinusoid(Record):

    def __init__(self, dim, cap= None, name= 'sinusoid'):
        self.dim = dim
        self.name = name
        with scope(name):
            self.pos = tf.constant(sinusoid(dim, cap, array= True), tf.float32) if cap else None

    def __call__(self, time, name= None):
        with scope(name or self.name):
            return sinusoid(self.dim, time) if self.pos is None else self.pos[:,:time]


class MlpBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.lin  = Conv(4*dim, dim, name= 'lin')
            self.lex  = Conv(dim, 4*dim, name= 'lex')
            self.norm = Normalize(dim)

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.lex(tf.nn.relu(self.lin(x)))))


class AttBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.att  = Attention(dim)
            self.norm = Normalize(dim)

    def __call__(self, x, v, m, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.att(x, v, m)))


class Encode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         MlpBlock(dim, 'm2') \
                # ,         AttBlock(dim, 's3') \
                # ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   's' == btype: x = block(x, x, m, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown encode block')
            return x


class Decode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         AttBlock(dim, 'a1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         AttBlock(dim, 'a2') \
                ,         MlpBlock(dim, 'm2') \
                # ,         AttBlock(dim, 's3') \
                # ,         AttBlock(dim, 'a3') \
                # ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         AttBlock(dim, 'a4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         AttBlock(dim, 'a5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         AttBlock(dim, 'a6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, w, n, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   's' == btype: x = block(x, x, m, dropout)
                elif 'a' == btype: x = block(x, w, n, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown decode block')
            return x


class Model(Record):
    """-> Record

    model = Model.new( ... )
    train = model.data( ... ).train( ... )
    valid = model.data( ... ).valid( ... )
    infer = model.data( ... ).infer( ... )

    """
    _new = 'dim_emb', 'dim_mid', 'dim_src', 'dim_tgt', 'cap', 'eos', 'bos'

    @staticmethod
    def new(src_dwh, tgt_dwh, dim_emb= 256, cap= 256, eos= 1, bos= 2):
        """-> Model with fields

          decode : Decode
          encode : Encode
         emb_tgt : Embed
         emb_src : Embed

        """
        assert not dim_emb % 2
        dim_src, w_s, h_s = src_dwh
        dim_tgt, w_t, h_t = tgt_dwh
        return Model(
              decode= Decode(dim_emb, name= 'decode')
            , encode= Encode(dim_emb, name= 'encode')
            , emb_tgt= Conv(dim_emb, w_t*h_t, name= 'emb_tgt')
            , emb_src= Conv(dim_emb, w_s*h_s, name= 'emb_src')
            , out_img= Conv(w_t*h_t, dim_emb, name= 'out_img')
            # , out_idx= Conv(dim_tgt, dim_emb, name= 'out_idx')
            , dim_emb= dim_emb
            , dim_tgt= dim_tgt
            , w_s= w_s, h_s= h_s
            , w_t= w_t, h_t= h_t
            , bos= bos
            , eos= eos
            , cap= cap + 1)

    def data(self, src_img= None, len_src= None, tgt_img= None, len_tgt= None):
        """-> Model with new fields

        position : Sinusoid
            src_ : i32 (b, ?)     source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)     target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)     source with `eos` trimmed among the batch
             tgt : i32 (b, t)     target with `eos` trimmed among the batch, padded with `bos`
            mask : b8  (b, t)     target sequence mask
            true : i32 (?,)       target references
         max_tgt : i32 ()         maximum target length
         max_src : i32 ()         maximum source length
        mask_tgt : f32 (1, t, t)  target attention mask
        mask_src : f32 (b, 1, s)  source attention mask

        """
        src_img = placeholder(tf.uint8, (None, None, self.h_s, self.w_s), src_img, 'src_img') # b s h w
        tgt_img = placeholder(tf.uint8, (None, None, self.h_t, self.w_t), tgt_img, 'tgt_img') # b t h w
        # tgt_idx = placeholder(tf.int32, (None, None                    ), tgt_idx, 'tgt_idx') # b t
        len_src = placeholder(tf.int32, (None,                         ), len_src, 'len_src')
        len_tgt = placeholder(tf.int32, (None,                         ), len_tgt, 'len_tgt')
        with scope('src'):
            src = tf.to_float(flatten(src_img, 2, 3)) / 255.0 # b s c <- b s h w
            max_src = tf.reduce_max(len_src)
            mask_src = tf.log(tf.expand_dims(tf.sequence_mask(len_src, dtype= tf.float32), axis= 1))
        with scope('tgt'):
            tgt = tf.to_float(flatten(tgt_img, 2, 3)) / 255.0 # b t c <- b t h w
            max_tgt = tf.reduce_max(len_tgt) + 1
            mask_tgt = tf.log(tf.expand_dims(causal_mask(max_tgt), axis= 0))
            mask = tf.sequence_mask(len_tgt + 1, dtype= tf.float32)
            btru = tf.pad(tgt, ((0,0),(1,0),(0,0)), constant_values= 0.0)
            true = tf.pad(tgt, ((0,0),(0,1),(0,0)), constant_values= 1.0)
            # tidx = tf.pad(tgt_idx, ((0,0),(0,1)), constant_values= 1)
            true, tgt = tf.boolean_mask(true, mask), btru
        return Model(
            position= Sinusoid(self.dim_emb, self.cap)
            , src_img= src_img, len_src= len_src, mask_src= mask_src, max_src= max_src, src= src
            , tgt_img= tgt_img, len_tgt= len_tgt, mask_tgt= mask_tgt, max_tgt= max_tgt, tgt= tgt
            , true= true, mask= mask
            , **self)

    def infer(self):
        """-> Model with new fields, autoregressive

        len_tgt : i32 ()      steps to unfold aka t
           pred : i32 (b, t)  prediction, hard

        """
        dropout = identity
        with scope('infer'):
            with scope('encode'):
                w = self.position(self.max_src) + self.emb_src(tf.transpose(self.src, (0,2,1)))
                w = self.encode(w, self.mask_src, dropout) # bds
            with scope('decode'):
                # todo fixme
                cap = placeholder(tf.int32, (), self.cap)
                msk = tf.log(tf.expand_dims(causal_mask(cap), axis= 0)) # 1tt
                pos = self.position(cap) # dt
                i,q = tf.constant(0), tf.zeros_like(self.src[:,:1]) + self.bos
                def body(i, q):
                    j = i + 1
                    x = pos[:,:j] + self.emb_tgt(q) # bdj <- bj
                    x = self.decode(x, msk[:,:j,:j], w, self.mask_src, dropout) # bdj
                    p = tf.expand_dims( # b1
                        tf.argmax( # b
                            self.emb_tgt( # bn
                                tf.squeeze( # bd
                                    x[:,:,-1:] # bd1 <- bdj
                                    , axis= -1))
                            , axis= -1, output_type= tf.int32)
                        , axis= -1)
                    return j, tf.concat((q, p), axis= -1) # bk <- bj, b1
                cond = lambda i, q: ((i < cap) & ~ tf.reduce_all(tf.equal(q[:,-1], self.eos)))
                _, p = tf.while_loop(cond, body, (i, q), back_prop= False, swap_memory= True)
                pred = p[:,1:]
        return Model(self, len_tgt= cap, pred= pred)

    def valid(self, dropout= identity, smooth= None):
        """-> Model with new fields, teacher forcing

           output : f32 (?, dim_tgt)  prediction on logit scale
             prob : f32 (?, dim_tgt)  prediction, soft
             pred : i32 (?,)          prediction, hard
        errt_samp : f32 (?,)          errors
        loss_samp : f32 (?,)          losses
             errt : f32 ()            error rate
             loss : f32 ()            mean loss

        """
        with scope('emb_src_'): w = self.position(self.max_src) + dropout(self.emb_src(tf.transpose(self.src, (0,2,1)))) # bcs
        with scope('emb_tgt_'): x = self.position(self.max_tgt) + dropout(self.emb_tgt(tf.transpose(self.tgt, (0,2,1)))) # bct
        w = self.encode(w, self.mask_src,                   dropout, name= 'encode_') # bds
        x = self.decode(x, self.mask_tgt, w, self.mask_src, dropout, name= 'decode_') # bdt
        with scope('output_'):
            y = tf.boolean_mask(tf.transpose(self.out_img(x), (0,2,1)), self.mask) # ?d <- btd <- bdt
            # z = self.out_idx(x)
        with scope('pred_'): pred = tf.clip_by_value(y, 0.0, 1.0)
        # with scope('prob_'): prob = tf.nn.softmax(z, axis= -1)
        # with scope('pidx_'): pidx = tf.argmax(z, axis= -1, output_type= tf.int32)
        with scope('loss_'):
            diff = self.true - pred
            mae = tf.reduce_mean(tf.abs(diff), axis= -1)
            mse = tf.reduce_mean(tf.square(diff), axis= -1)
            # xid = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels= self.tidx)
            # err = tf.not_equal(self.tidx, pidx)
            loss = tf.reduce_mean(mae)
        return Model(self, pred= pred, mae= mae, mse= mse, loss= loss)

    def train(self, dropout= 0.1, smooth= 0.1, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Model with new fields, teacher forcing

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        along with all the fields from `valid`

        """
        dropout, smooth = Dropout(dropout, (None, self.dim_emb, None)), Smooth(smooth, self.dim_tgt)
        self = self.valid(dropout= dropout, smooth= smooth)
        with scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (self.dim_emb ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Model(self, dropout= dropout, smooth= smooth, step= s, lr= lr, down= up)


def batch_run(sess, model, fetch, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetch, feed)


if '__main__' == __name__:

    trial = 'tsae'
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
        return cws(src) + cwt(tgt)

    def batch(src= src_train, tgt= tgt_train, size= 128, seed= 0):
        for bat in batch_sample(len(tgt), size, seed):
            yield feed(src[bat], tgt[bat])

    model = Model.new(cws.dwh(), cwt.dwh())
    train = model.data(*pipe(batch, (tf.uint8, tf.int32, tf.uint8, tf.int32))).train()
    valid = model.data().valid()
    dummy = tuple(placeholder(tf.float32, ()) for _ in range(2))

    def log(step
            , wtr= tf.summary.FileWriter("/cache/tensorboard-logdir/lepidodendron/{}".format(trial))
            , log= tf.summary.merge(
                (  tf.summary.scalar('step_mae', dummy[0])
                 , tf.summary.scalar('step_mse', dummy[1])))
            , fet= (valid.mae, valid.mse)
            , inp= (valid.src_img, valid.len_src, valid.tgt_img, valid.len_tgt)
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
