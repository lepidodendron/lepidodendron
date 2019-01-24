from PIL import Image, ImageFont, ImageDraw
from collections import Counter
from util import Record
from util_io import load_txt, load_pkl, save_pkl
from util_np import np, vpack


def chars(lines, coverage= 0.9995):
    char2freq = Counter(char for line in lines for char in line)
    total = sum(char2freq.values())
    cover = char2freq[" "]
    del char2freq[" "]
    chars = [" "]
    for char, freq in char2freq.most_common():
        if coverage <= (cover / total): break
        chars.append(char)
        cover += freq
    return "".join(chars)


def write(char, font, width, height, offset, mode= 'L', fill= 255):
    image = Image.new(mode, (width, height))
    ImageDraw.Draw(image).text((0, offset), char, fill= fill, font= font)
    return np.array(image.getdata(), dtype= np.uint8).reshape(height, width)


class CharWright(Record):

    @staticmethod
    def new(chars, font= '../data/NotoSansMono-Regular.ttf', size= 20):
        font = ImageFont.truetype(font, size)
        width, height = map(max, zip(*map(font.getsize  , chars)))
        _    , offset = map(min, zip(*map(font.getoffset, chars)))
        height -= offset
        offset = -offset
        imags = np.stack([write(char, font, width, height, offset) for char in chars])
        return CharWright(
              chars= chars, char2rank= {char: rank for rank, char in enumerate(chars)}
            , imags= imags, char2imag= dict(zip(chars, imags))
            , offset= offset
            , height= height
            , width= width
            , font= font
            , size= size)

    @staticmethod
    def load(path):
        return CharWright.new(**load_pkl(path))

    def save(self, path):
        save_pkl(path, {'chars': self.chars, 'font': self.font.path, 'size': self.size})

    def write1(self, line):
        image = []
        for char in line:
            try:
                imag = self.char2imag[char]
            except KeyError:
                imag = write(char, self.font, self.width, self.height, self.offset)
            finally:
                image.append(imag)
        return np.stack(image, axis= 1)

    def write(self, lines, maxlen):
        images = [self.write1(line[:maxlen]) for line in lines]
        return np.stack([np.pad(htw, ((0,0),(0,maxlen-htw.shape[1]),(0,0)), 'constant') for htw in images])

    def index1(self, line):
        # index = rank + 1
        # index 0 (rank -1) is reserved for unknown char
        return 1 + np.array([self.char2rank.get(char, -1) for char in line], np.int32)

    def index(self, lines, maxlen):
        # index 1 (rank 0) aka the whitespace is used for padding
        indexs = [self.index1(line[:maxlen]) for line in lines]
        return vpack(indexs, (len(indexs), maxlen), 1, np.int32)

    def __call__(self, lines):
        length = np.fromiter(map(len, lines), np.int32, len(lines))
        maxlen = length.max()
        return self.write(lines, maxlen), self.index(lines, maxlen), length

    def nchars(self):
        return 1 + len(self.chars)

    def tostr1(self, idx):
        chars = []
        for r in idx-1:
            if 0 <= r:
                char = self.chars[r]
            else:
                char = "�"
            chars.append(char)
        return "".join(chars)

    def tostr(self, idxs):
        for idx in idxs:
            yield self.tostr1(idx)


# import matplotlib.pyplot as plt
# def plot(x):
#     if 3 == x.ndim:
#         h, t, w = x.shape
#         x = x.reshape(h, t*w)
#     elif 4 == x.ndim:
#         n, h, t, w = x.shape
#         x = x.reshape(n*h, t*w)
#     plt.imshow(x, cmap= 'gray')
#     plt.show()
