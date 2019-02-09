#!/usr/bin/env python3

from util_cw import chars, CharWright
from util_io import load_txt, save_txt
from util_np import np

path_src = "../data/UNv1.0.en-zh.zh"
path_tgt = "../data/UNv1.0.en-zh.en"
max_char = 128

#############
# load data #
#############

src_tgt = []
for src, tgt in zip(load_txt(path_src), load_txt(path_tgt)):
    src = src.strip()
    tgt = tgt.strip()
    if 3 <= len(src) <= max_char and 3 <= len(tgt) <= max_char:
        src_tgt.append((src, tgt))

np.random.seed(0)
np.random.shuffle(src_tgt)

src, tgt = zip(*src_tgt)
del src_tgt

#############
# save data #
#############

cws = CharWright.new(chars(src), font= "../data/NotoSansMonoCJKsc-Regular.otf")
cwt = CharWright.new(chars(tgt))

cws.save("../data/zh-en/cws.pkl")
cwt.save("../data/zh-en/cwt.pkl")

save_txt("../data/zh-en/src.txt", src)
save_txt("../data/zh-en/tgt.txt", tgt)
