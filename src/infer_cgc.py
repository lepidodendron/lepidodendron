from model_cgc import model
ckpt = "cgc_36"
mode = 3

from infer import infer, trim_str
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

def translate(src, mode):
    for i, j in partition(len(src), 256):
        src_idx, len_src = cws(src[i:j], ret_img= False, ret_idx= True)
        pred, pidx = infer(mode, m, sess, cwt, src_idx, len_src)
        yield from trim_str(pidx, cwt)

save_txt("../tmp/prd", translate(src, mode))
save_txt("../tmp/tgt", tgt)

# sacrebleu -tok intl -b -i ../tmp/prd ../tmp/tgt
