from PIL import Image, ImageFont, ImageDraw
from collections import Counter
from util import Record
from util_io import load_txt, load_pkl, save_pkl
from util_np import np


def chars(lines, coverage= 0.9995):
    char2freq = Counter(char for line in lines for char in line)
    chars = []
    cover, total = 0, sum(char2freq.values())
    for char, freq in char2freq.most_common():
        cover += freq
        if coverage <= (cover / total): break
        chars.append(char)
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
                self.char2imag[char] = imag
            finally:
                image.append(imag)
        return np.stack(image, axis= 1)

    def write(self, lines):
        images = tuple(map(self.write1, lines))
        length = np.array([image.shape[1] for image in images])
        packed = np.stack([np.pad(i, ((0,0),(0,p),(0,0)), 'constant') for i, p in zip(images, length.max() - length)])
        return packed, length


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
