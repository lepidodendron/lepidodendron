from itertools import islice
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def write(text, font= ImageFont.truetype('../data/NotoSansMono-Regular.ttf', 20), height= 27, width= 12):
    if isinstance(text, str): text = text.split("\n")
    lengths = np.fromiter(map(len, text), np.int32, len(text))
    nrows = lengths.size * height
    ncols = lengths.max() * width
    image = Image.new('L', (ncols, nrows))
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(text):
        draw.text((0, row * height), line, fill= 255, font= font)
    return np.array(image.getdata(), dtype= np.uint8).reshape(nrows, ncols), lengths


def vpack(arrays, shape, fill, dtype= None):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.

    """
    array = np.full(shape, fill, dtype)
    for row, arr in zip(array, arrays):
        row[:len(arr)] = arr
    return array


def partition(n, m, discard= False):
    """yields pairs of indices which partitions `n` nats by `m`.  if not
    `discard`, also yields the final incomplete partition.

    """
    steps = range(0, 1 + n, m)
    yield from zip(steps, steps[1:])
    if n % m and not discard:
        yield n - (n % m), n


def sample(n, seed= 0):
    """yields samples from `n` nats."""
    data = list(range(n))
    while True:
        np.random.seed(seed)
        np.random.shuffle(data)
        yield from data


def batch_sample(n, m, seed= 0):
    """yields `m` samples from `n` nats."""
    stream = sample(n, seed)
    while True:
        yield np.fromiter(stream, np.int, m)


def batch(stream, batch, shuffle= 2**14, seed= 0, discard= True):
    """yields batches from `stream`."""
    assert not shuffle % batch
    while True:
        buf = list(islice(stream, shuffle))
        if not buf: break
        np.random.seed(seed)
        np.random.shuffle(buf)
        yield from (buf[i:j] for i, j in partition(len(buf), batch, discard= discard))
