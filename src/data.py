#!/usr/bin/env python3

from util_cw import chars, CharWright
from util_io import load_txt, save_txt
from util_np import np

path_src = "../data/europarl-v7.de-en.de"
path_tgt = "../data/europarl-v7.de-en.en"
max_char = 256

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

cw_src = CharWright.new(chars(src))
cw_tgt = CharWright.new(chars(tgt))

cw_src.save("../data/cw_src.pkl")
cw_tgt.save("../data/cw_tgt.pkl")

save_txt("../data/src.txt", src)
save_txt("../data/tgt.txt", tgt)
