from model_cgg import model
ckpt = "cgg_36"

from itertools import islice
from util_cw import CharWright
from util_io import load_txt, save_txt
from util_np import np, partition
from util_tf import tf
sess = tf.InteractiveSession()

# load model
cws = CharWright.load("../data/cws.pkl")
cwt = CharWright.load("../data/cwt.pkl")
m = model('infer', cws.dwh(), cwt.dwh())
saver = tf.train.Saver()
saver.restore(sess, "../ckpt/{}".format(ckpt))

# the first 4096 instances are used for validation
src = np.array(list(islice(load_txt("../data/src.txt"), 4096)))
tgt = np.array(list(islice(load_txt("../data/tgt.txt"), 4096)))
val = np.array(sorted(range(len(src)), key= lambda i: len(src[i])))
src = src[val]
tgt = tgt[val]

def autoreg(src_idx, len_src, max_len= 256, eos= 1):
    emb_src = sess.run(m.emb_src, {m.src_idx: src_idx, m.len_src: len_src})
    x_shape = 1, len_src.size, cwt.height * cwt.width
    x = np.zeros(x_shape, np.uint8)
    s = sess.run(m.state_in, {m.fire: x})
    idxs = []
    for _ in range(max_len):
        s, pred, pidx = sess.run((m.state_ex, m.pred, m.pidx), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
        if np.all(pidx == eos): break
        idxs.append(pidx)
        x = pred
        x.shape = x_shape
    return np.concatenate(idxs).T

def trim_str(idxs):
    for s in cwt.string(idxs).split("\n"):
        try:
            s = s[:s.index("█")]
        except ValueError:
            pass
        yield s

def translate(src):
    for i, j in partition(len(src), 256):
        yield from trim_str(autoreg(*cws(src[i:j], ret_img= False, ret_idx= True)))

save_txt("../tmp/prd", translate(src))
save_txt("../tmp/tgt", tgt)

# sacrebleu -tok intl -b -i ../tmp/prd ../tmp/tgt
