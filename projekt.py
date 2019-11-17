import sys, time, math

import numpy as np
from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_yen, gaussian, median, threshold_li, threshold_isodata, threshold_otsu, threshold_mean
from skimage.feature import canny
from skimage.morphology import remove_small_objects, binary_opening, remove_small_holes, dilation, disk
from skimage.color import rgb2hsv

from skimage.measure import label, regionprops


HEIGHT = 250
WIDTH = 250

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


def im_resize(img):
    return resize(img, (HEIGHT, WIDTH))


def get_contrasted(img, perc):
    # MIN = np.percentile(img, perc)
    # MAX = np.percentile(img, 100 - perc)
    # norm = (img - MIN) / (MAX - MIN)
    # norm[norm[:, :] > 1] = 1
    # norm[norm[:, :] < 0] = 0

    # OR

    gamma_coef = 2.5
    norm = img**gamma_coef

    return norm


def main(level='easy', img_number=3):
    for n in range(1, 1 + img_number):
        # read image from input and make edge detection to find ellipse next
        im = im_resize(io.imread(f'examples/{level}/coin{n}.jpg', as_gray=True))

        # plt.imshow(1 - im, cmap='binary')
        # plt.title("cleared1")
        # plt.show()
        # time.sleep(1)

        im = median(im, disk(3))

        # add contrast
        im = get_contrasted(im, 15)

        # plt.imshow(1-im, cmap='binary')
        # plt.title("contrasted")
        # plt.show()
        # time.sleep(1)

        im = median(im, disk(3))

        thresh = threshold_isodata(im)
        binary = im > thresh

        # plt.imshow(1-binary, cmap='binary')
        # plt.title("Afterthrashed")
        # plt.show()
        # time.sleep(1)

        binary_cleaned = remove_small_holes(binary, 1000)
        binary_cleaned = remove_small_objects(binary_cleaned, 1000)

        # plt.imshow(1-binary_cleaned, cmap='binary')
        # plt.title("Cleaned")
        # plt.show()
        # time.sleep(1)

        binary_smooth = medfilt(binary_cleaned, 3)
        edges = dilation(canny(binary_smooth))

        # plt.imshow(1-binary_smooth, cmap='binary')
        # plt.title("Smooth")
        # plt.show()
        # time.sleep(1)

        labels = label(edges)

        im_rgb = im_resize(io.imread(f'examples/{level}/coin{n}.jpg'))

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(im_rgb)

        for region in regionprops(labels):

            # th = 4 * math.pi * region.area / (region.perimeter**2)
            # print(th, region.area, region.perimeter)
            # take regions with large enough areas
            if region.area >= 50:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='green', linewidth=2)
                center = [int(rect.get_x() + rect.get_width() / 2), int(rect.get_y() + rect.get_height() / 2), ]
                # m = min([rect.get_width(), rect.get_height()])
                col = np.average(rgb2hsv(im_rgb)[center[1] - 5:center[1] + 5, center[0] - 5:center[0] + 5, :], axis=(0, 1))
                h, s, v = col[0], col[1], col[2]

                if v < .1:
                    print("Black")
                    continue
                # detect coin's color
                if .7 > s > .3:
                    txt = 'Golden'
                elif .3 > s > 0:
                    txt = 'Silver'
                else:
                    continue
                ax.add_patch(rect)
                ax.text(rect.get_x(), rect.get_y() - 5, txt, fontdict=font)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()


def mainloop():
    # main('easy', 3)
    main("medium", 2)
    # main("hard", 2)


if __name__ == '__main__':
    mainloop()
