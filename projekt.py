import sys, time, math, os

import numpy as np
from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt
from skimage import io
from skimage.transform import resize
from skimage.filters import sobel, threshold_yen, gaussian, median, threshold_li, threshold_isodata, threshold_otsu, threshold_mean
from skimage.feature import canny
from skimage.morphology import watershed, remove_small_objects, binary_opening, remove_small_holes, dilation, disk, binary_closing, erosion
from skimage.color import rgb2hsv
from skimage import exposure
from scipy import ndimage as ndi

from skimage.measure import label, regionprops

MAX_WIDTH = 500

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 12,
        }


def im_resize(img):
    return resize(img, (int(img.shape[0]*(MAX_WIDTH/img.shape[1])), MAX_WIDTH))


def get_contrasted(img, perc):
    # MIN = 180/255
    # MAX = 1
    # norm = (img - MIN) / (MAX - MIN)
    # norm[norm[:, :] > 1] = 1
    # norm[norm[:, :] < 0] = 0

    # OR
    #
    gamma_coef = 2
    norm = img**gamma_coef

    return norm


def check_color(h, s, v):
    txt = None
    if v < .1:
        print('Too dark')
    # detect coin's color
    if .7 > s > .3 and 65/360 > h > 20/360:
        txt = 'Golden'
    elif .3 >= s > 0:
        txt = 'Silver'
    else:
        print(f"Unknown color, hue: {h*360}, saturation: {s}")
    return txt


def get_average_hsv(segment):
    col = np.average(rgb2hsv(segment), axis=(0, 1))
    h, s, v = col[0], col[1], col[2]
    return h, s, v


def check_if_five(coin):
    # -- diameter for golden kernel is 16, whole coin diameter is 24 (1/3)
    r = coin.shape[0]//6
    rf = r//2
    c = coin.shape[1]//2
    coin_check_point = coin[rf:rf+rf, c-rf:c+rf, :]
    h, s, v = get_average_hsv(coin_check_point)
    if check_color(h, s, v) == "Silver":
        return True
    return False


def check_if_two(coin):
    r = coin.shape[0] // 5
    rf = r // 3
    c = coin.shape[1] // 2
    coin_check_point = coin[rf:rf + rf, c - rf:c + rf, :]
    print(coin_check_point.shape, r, rf, c)
    h, s, v = get_average_hsv(coin_check_point)
    if check_color(h, s, v) == "Golden":
        return True
    return False


def main(level='easy'):
    for n, file in enumerate(os.listdir(f'examples/{level}')):
        if not any([extension in file.lower() for extension in ['jpg', 'jpeg', 'png']]):
            continue
        # if n != 2:
        #     continue
        # -- read image from input and make edge detection to find ellipse next
        im = im_resize(io.imread(f'examples/{level}/{file}', as_gray=True))

        # plt.imshow(im, cmap='gray')
        # plt.title("image")
        # plt.show()
        # time.sleep(1)

        # im = median(im, disk(3))

        # -- add contrast
        im = get_contrasted(im, 15)

        # plt.imshow(im, cmap='gray')
        # plt.title("contrasted")
        # plt.show()
        # time.sleep(1)

        # im = median(im, disk(2))
        #
        # thresh = threshold_yen(im)
        # binary = im > thresh
        #
        # plt.imshow(binary, cmap='gray')
        # plt.title("Afterthrashed")
        # plt.show()
        # time.sleep(1)
        #
        # binary_cleaned = remove_small_holes(binary, 100)
        # binary_cleaned = remove_small_objects(binary_cleaned, 100)
        #
        # plt.imshow(1-binary_cleaned, cmap='binary')
        # plt.title("Cleaned")
        # plt.show()
        # time.sleep(1)
        #
        # binary_smooth = medfilt(binary_cleaned, 3)
        # edges = dilation(canny(binary_smooth))
        #
        # plt.imshow(edges, cmap='gray')
        # plt.title("Smooth")
        # plt.show()
        # time.sleep(1)

        im = binary_closing(canny(im))

        # plt.imshow(im, cmap='gray')
        # plt.title("Canny")
        # plt.show()
        # time.sleep(1)

        objects = binary_closing(ndi.binary_fill_holes(im))
        # objects = remove_small_holes(objects, 500)

        # plt.imshow(objects, cmap='gray')
        # plt.title("Objects small holes")
        # plt.show()
        # time.sleep(1)

        objects = remove_small_objects(objects, 500)

        # plt.imshow(objects_cleaned, cmap='gray')
        # plt.title("Objects")
        # plt.show()
        # time.sleep(1)

        labels = label(objects)

        im_rgb = im_resize(io.imread(f'examples/{level}/{file}'))

        # plt.imshow(im_rgb)
        # plt.title("img")
        # plt.show()
        # time.sleep(1)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(im_rgb)

        for region in regionprops(labels):

            # th = 4 * math.pi * region.area / (region.perimeter**2)
            # print(th, region.area, region.perimeter)
            # take regions with large enough areas
            if region.area >= 40:
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
                elif coin_center_color == "Silver":
                    two = check_if_two(im_rgb[minr:maxr, minc:maxc, :])
                    if two:
                        coin_center_color = "Dwa złote"
                ax.add_patch(rect)
                ax.text(rect.get_x(), rect.get_y() - 2, coin_center_color, fontdict=font)
                ax.axis('off')
        plt.show()


def mainloop():
    main('easy')
    # main("medium")
    # main("hard")


if __name__ == '__main__':
    mainloop()
