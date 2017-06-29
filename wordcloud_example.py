import jieba
from jieba.analyse import extract_tags
from wordcloud import WordCloud
from os import path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class WordCloudS(object):
    def __init__(self, text_file_path, font_path=None, mask_path=None):
        self.text_file_path = text_file_path
        self.font_path = font_path if font_path else u'./static/simhei.ttf'
        self.mask_path = mask_path
        self.words = []

    @property
    def get_words(self):
        with open(self.text_file_path, encoding='utf-8') as f:
            for line in f.readlines():
                self.words.extend(extract_tags(line))

        self.words = ' '.join(self.words)
        return self.words

    def show(self):
        mask_img = None
        if self.mask_path:
            mask_img = np.array(Image.open(self.mask_path))

        my_wordcloud = WordCloud(font_path=self.font_path,
            background_color="white",
            margin=5,
            width=1800,
            height=800,
            mask=mask_img,
            )

        my_wordcloud = my_wordcloud.generate(self.get_words)

        plt.figure()
        plt.imshow(my_wordcloud)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    d = path.dirname(path.abspath(__file__))
    text_path = path.join(d, 'res/answer_spam.txt')
    font_path = path.join(d, 'static/simhei.ttf')
    mask_path = path.join(d, 'static/bird_mask.png')
    ge = WordCloudS(text_path, font_path, mask_path)
    ge.show()

