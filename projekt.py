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
from skimage import exposure

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
    # MIN = 180/255
    # MAX = 1
    # norm = (img - MIN) / (MAX - MIN)
    # norm[norm[:, :] > 1] = 1
    # norm[norm[:, :] < 0] = 0

    # OR
    #
    gamma_coef = 2.5
    norm = img**gamma_coef

    return norm


def check_color(h, s, v):
    txt = None
    if v < .1:
        print('Too dark')
    # detect coin's color
    if .7 > s > .3 and 55/360 > h > 20/360:
        txt = 'Golden'
    elif .3 > s > 0:
        txt = 'Silver'
    else:
        print(h*360)
    return txt


def get_average_hsv(segment):
    col = np.average(rgb2hsv(segment), axis=(0, 1))
    h, s, v = col[0], col[1], col[2]
    return h, s, v


def check_if_five(coin):
    # -- diameter for golden kernel is 16, whole coin diameter is 24 (1/3)
    r = coin.shape[0]//8
    rf = r//3
    c = coin.shape[1]//2
    coin_check_point = coin[rf:rf+rf, c-rf:c+rf, :]
    h, s, v = get_average_hsv(coin_check_point)
    if check_color(h, s, v) == "Silver":
        return True
    return False


def main(level='easy', img_number=3):
    for n in range(1, 1 + img_number):
        # if n != 2:
        #     continue
        # -- read image from input and make edge detection to find ellipse next
        im = im_resize(io.imread(f'examples/{level}/coin{n}.jpg', as_gray=True))

        im = exposure.rescale_intensity(im)

        # plt.imshow(1 - im, cmap='binary')
        # plt.title("cleared1")
        # plt.show()
        # time.sleep(1)

        im = median(im, disk(3))

        # -- add contrast
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

        # plt.imshow(im_rgb)
        # plt.title("img")
        # plt.show()
        # time.sleep(1)


        fig, ax = plt.subplots(figsize=(6, 6))

        # im_hsv = rgb2hsv(im_rgb)

        # print(np.max(im_hsv[:, :, 1]))

        # mask_golden = ((im_hsv[:, :, 1] > .3) & (im_hsv[:, :, 1] > .7))
        # mask_golden = remove_small_holes(mask_golden, 1000)
        # mask_golden = remove_small_objects(mask_golden, 1000)
        #
        #
        # plt.imshow(mask_golden, cmap='gray')
        # plt.title("golden")
        # plt.show()
        # time.sleep(1)
        #
        # mask_silver = im_hsv[:, :, 1] < .3
        # mask_silver = remove_small_holes(mask_silver, 1000)
        # mask_sliver = remove_small_objects(mask_silver, 1000)
        #
        # plt.imshow(mask_sliver, cmap='gray')
        # plt.title("silver")
        # plt.show()
        # time.sleep(1)

        # continue

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

                h, s, v = get_average_hsv(im_rgb[center[1] - 5:center[1] + 5, center[0] - 5:center[0] + 5, :])
                coin_center_color = check_color(h, s, v)
                if isinstance(coin_center_color, type(None)):
                    continue
                if coin_center_color == "Golden":
                    five = check_if_five(im_rgb[minr:maxr, minc:maxc, :])
                    if five:
                        coin_center_color = "Pięć złotych"
                ax.add_patch(rect)
                ax.text(rect.get_x(), rect.get_y() - 5, coin_center_color, fontdict=font)
                ax.axis('off')
        plt.show()


def mainloop():
    # main('easy', 4)
    main("medium", 4)
    # main("hard", 2)


if __name__ == '__main__':
    mainloop()
