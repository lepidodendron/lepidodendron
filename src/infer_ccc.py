from model_ccc import model
ckpt = "ccc_18"

from infer import trim_str
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

def infer(src_idx, len_src, t= 256, bos= 2):
    emb_src = sess.run(m.emb_src, {m.src_idx: src_idx, m.len_src: len_src})
    x = np.full((1, len_src.size), bos, np.int32)
    s = sess.run(m.state_in, {m.fire: x})
    idxs = []
    for _ in range(t):
        s, x = sess.run((m.state_ex, m.pidx), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
        idxs.append(x)
    return np.concatenate(idxs).T

def infer5(src_idx, len_src, t= 256, bos= 2):
    emb_src = sess.run(m.emb_src, {m.src_idx: src_idx, m.len_src: len_src})
    x = np.full((1, len_src.size), bos, np.int32)
    s, x = sess.run((m.state_in, m.fire2), {m.fire: x})
    idxs = []
    for _ in range(t):
        s, x = sess.run((m.state_ex, m.prob), {m.state_in: s, m.fire2: x, m.emb_src: emb_src, m.len_src: len_src})
        idxs.append(np.argmax(x, axis= -1))
    return np.concatenate(idxs).T

def translate(src):
    for i, j in partition(len(src), 256):
        src_idx, len_src = cws(src[i:j], ret_img= False, ret_idx= True)
        yield from trim_str(infer(src_idx, len_src), cwt)

save_txt("../tmp/prd", translate(src))
save_txt("../tmp/tgt", tgt)

# sacrebleu -tok intl -b -i ../tmp/prd ../tmp/tgt
