from PIL import Image, ImageFont, ImageDraw
from collections import Counter
from util import Record
from util_io import load_txt, load_pkl, save_pkl
from util_np import np, vpack


def chars(lines, coverage= 0.9995):
    """returns a string containing the most frequent types of characters
    which cover the strings in `lines`, ordered by their ranks.

    rank 0 is rigged to always be the whitespace.

    """
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


def pad(lines, char= "█"):
    maxlen = max(map(len, lines))
    return [(line + char * (maxlen - len(line))) for line in lines]


class CharWright(Record):

    @staticmethod
    def new(chars, font= '../data/NotoSansMono-Regular.ttf', size= 20):
        assert "�" not in chars # unk
        assert "█" not in chars # eos
        font = ImageFont.truetype(font, size)
        width, height = map(max, zip(*map(font.getsize  , chars)))
        _    , offset = map(min, zip(*map(font.getoffset, chars)))
        height -= offset
        offset = -offset
        chars = "�█" + chars
        char2idx = {char: rank for rank, char in enumerate(chars)}
        char2img = {char: write(char, font, width, height, offset) for char in chars}
        char2img["█"] = char2img["█"] * 0 + 255
        return CharWright(
            chars= chars
            , char2idx= char2idx
            , char2img= char2img
            , offset= offset
            , height= height
            , width= width
            , font= font
            , size= size)

    @staticmethod
    def load(path):
        return CharWright.new(**load_pkl(path))

    def save(self, path):
        save_pkl(path, {'chars': self.chars[2:], 'font': self.font.path, 'size': self.size})

    def write1(self, line, nrow= 1):
        image = []
        for char in line:
            try:
                image.append(self.char2img[char])
            except KeyError:
                image.append(write(char, self.font, self.width, self.height, self.offset))
        return np.stack(image).reshape(nrow, -1, self.height, self.width)

    def write(self, lines):
        return self.write1("".join(pad(lines)), len(lines))

    def index1(self, line, nrow= 1):
        return np.array([self.char2idx.get(char, 0) for char in line], np.int32).reshape(nrow, -1)

    def index(self, lines):
        return self.index1("".join(pad(lines)), len(lines))

    def __call__(self, lines, ret_img= True, ret_idx= False, ret_len= True):
        line, nrow = "".join(pad(lines)), len(lines)
        ret = []
        if ret_img: ret.append(self.write1(line, nrow))
        if ret_idx: ret.append(self.index1(line, nrow))
        if ret_len: ret.append(np.fromiter(map(len, lines), np.int32, nrow))
        return tuple(ret)

    def dwh(self):
        return len(self.chars), self.width, self.height

    def string(self, idxs):
        return "\n".join(["".join([self.chars[i] for i in idx]) for idx in idxs])

    def _glyphs(self):
        glyphs = np.array([self.char2img[c] for c in self.chars], dtype= np.float32)
        glyphs /= 255.0
        glyphs.shape = len(self.chars), self.height * self.width
        return glyphs

    def match(self, image):
        if not hasattr(self, 'glyphs'): self.glyphs = self._glyphs()
        return np.argmin(np.mean(np.abs(self.glyphs - np.expand_dims(image, -2)), axis= -1), axis= -1)

    def average(self, probs):
        if not hasattr(self, 'glyphs'): self.glyphs = self._glyphs()
        return probs @ self.glyphs
