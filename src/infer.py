import matplotlib.pyplot as plt
import numpy as np


def autoreg1(m, sess, cwt, s, x, emb_src, len_src):
    s, pred, pidx = sess.run((m.state_ex, m.pred, m.pidx), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
    return s, pred, pidx

def autoreg2(m, sess, cwt, s, x, emb_src, len_src):
    s, pred = sess.run((m.state_ex, m.pred), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
    pidx = cwt.match(pred)
    return s, pred, pidx

def autoreg3(m, sess, cwt, s, x, emb_src, len_src):
    s, pidx = sess.run((m.state_ex, m.pidx), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
    pred = cwt.write(cwt.string(pidx)) / 255.0
    return s, pred, pidx

def autoreg4(m, sess, cwt, s, x, emb_src, len_src):
    s, pred = sess.run((m.state_ex, m.pred), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
    pidx = cwt.match(pred)
    pred = cwt.write(cwt.string(pidx)) / 255.0
    return s, pred, pidx

def autoreg5(m, sess, cwt, s, x, emb_src, len_src):
    s, prob = sess.run((m.state_ex, m.prob), {m.state_in: s, m.fire: x, m.emb_src: emb_src, m.len_src: len_src})
    pidx = np.argmax(prob, axis= -1)
    pred = cwt.average(prob)
    return s, pred, pidx

def infer(mode, m, sess, cwt, src_idx_or_img, len_src, t= 256):
    emb_src = sess.run(m.emb_src, {m.src_idx: src_idx_or_img, m.len_src: len_src} if 2 == np.ndim(src_idx_or_img) else {m.src_img: src_idx_or_img, m.len_src: len_src})
    autoreg = (autoreg1, autoreg2, autoreg3, autoreg4, autoreg5)[mode-1]
    n, (d, w, h) = len(emb_src), cwt.dwh()
    x_shape = 1, n, h * w
    x = np.zeros(x_shape, np.float32)
    s = sess.run(m.state_in, {m.fire: x})
    imgs, idxs = [], []
    for _ in range(t):
        s, pred, pidx = autoreg(m, sess, cwt, s, x, emb_src, len_src)
        imgs.append(pred)
        idxs.append(pidx)
        x = pred.reshape(x_shape)
    pred = np.concatenate(imgs).reshape(t, n, h, w).transpose(1, 0, 2, 3)
    pidx = np.concatenate(idxs).T
    return pred, pidx


def trim_str(idxs, cwt):
    for s in cwt.string(idxs).split("\n"):
        try:
            s = s[:s.index("█")]
        except ValueError:
            pass
        yield s


def zip_imgs(*imgs):
    # zips imgs of format nthw along h
    # all imgs must have the same n & w
    imgs = tuple(map(lambda x: x / x.max(), imgs))
    n, t, h, w = zip(*map(np.shape, imgs))
    assert 1 == len(set(n)) == len(set(w))
    n, t, h, w = n[0], max(t), sum(h), w[0]
    img = np.full((n, t, h, w), 1.0, dtype= np.float32)
    for i, rows in enumerate(zip(*imgs)):
        offset = 0
        for row in rows:
            t, h, _ = row.shape
            img[i,:t,offset:(offset+h)] = row
            offset += h
    return img


def plot(x):
    if 3 == x.ndim:
        t, h, w = x.shape
        x = np.transpose(x, (1, 0, 2)).reshape(h, t*w)
    elif 4 == x.ndim:
        n, t, h, w = x.shape
        x = np.transpose(x, (0, 2, 1, 3)).reshape(n*h, t*w)
    plt.imshow(x, cmap= 'gray')
    plt.show()
