import sys, time, math, os

import numpy as np
from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt
from skimage import io
from skimage.transform import resize
from skimage.filters import sobel, threshold_yen, gaussian, median, threshold_li, threshold_isodata, threshold_otsu, \
    threshold_mean
from skimage.feature import canny
from skimage.morphology import watershed, remove_small_objects, binary_opening, remove_small_holes, dilation, disk, \
    binary_closing, erosion
from skimage.color import rgb2hsv
from skimage import exposure
from scipy import ndimage as ndi

from skimage.measure import label, regionprops

# to resize image
MAX_WIDTH = 500

D1z = 23
D2z = 21.50
D5z = 24
D5 = 19.50
D10 = 16.50
D20 = 18.50

# ratios from https://www.nbp.pl/home.aspx?f=/banknoty_i_monety/monety_obiegowe/opisy.html
R_10_20 = {"f": "10 gr", "s": "20 gr", 'r': D10 / D20}
R_20_1z = {"f": "20 gr", "s": "1 zł", 'r': D20 / D1z}
R_10_1z = {"f": "10 gr", "s": "1 zł", 'r': D10 / D1z}
R_10_5 = {"f": "10 gr", "s": "5 gr", "r": D10 / D5}
R_20_5 = {"f": "20 gr", "s": "5 gr", "r": D20 / D5}
R_1z_5 = {"f": "1 zł", "s": "5 gr", "r": D1z / D5}
R_10_2z = {"f": "10 gr", "s": "2 zł", "r": D10 / D2z}
R_20_2z = {"f": "10 gr", "s": "2 zł", "r": D20 / D2z}
R_1z_2z = {"f": "10 gr", "s": "2 zł", "r": D1z / D2z}
R_10_5z = {"f": "10 gr", "s": "5 zł", "r": D10 / D5z}
R_20_5z = {"f": "10 gr", "s": "5 zł", "r": D20 / D5z}
R_1z_5z = {"f": "10 gr", "s": "5 zł", "r": D1z / D5z}



# font settings for label
font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 12,
        }


# resize image when upload
def im_resize(img):
    return resize(img, (int(img.shape[0] * (MAX_WIDTH / img.shape[1])), MAX_WIDTH))


#
def get_contrasted(img, perc):
    # MIN = 180/255
    # MAX = 1
    # norm = (img - MIN) / (MAX - MIN)
    # norm[norm[:, :] > 1] = 1
    # norm[norm[:, :] < 0] = 0

    # OR
    #
    gamma_coef = 1.5
    norm = img ** gamma_coef

    return norm


def check_color(h, s, v):
    txt = None
    if v < .1:
        print('Too dark')
    # detect coin's color
    if .7 > s > .3 and 65 / 360 > h > 20 / 360:
        txt = 'Golden'
    elif .3 >= s > 0:
        txt = 'Silver'
    else:
        print(f"Unknown color, hue: {h * 360}, saturation: {s}")
    return txt


def get_average_hsv(segment):
    col = np.average(rgb2hsv(segment), axis=(0, 1))
    h, s, v = col[0], col[1], col[2]
    return h, s, v


def check_if_five(coin):
    # -- diameter for golden kernel is 16, whole coin diameter is 24 (1/3)
    r = coin.shape[0] // 6
    rf = r // 2
    c = coin.shape[1] // 2
    coin_check_point = coin[rf:rf + rf, c - rf:c + rf, :]
    h, s, v = get_average_hsv(coin_check_point)
    if check_color(h, s, v) == "Silver":
        return True
    return False


def check_if_two(coin):
    r = coin.shape[0] // 5
    rf = r // 3
    c = coin.shape[1] // 2
    coin_check_point = coin[rf:rf + rf, c - rf:c + rf, :]
    h, s, v = get_average_hsv(coin_check_point)
    if check_color(h, s, v) == "Golden":
        return True
    return False


def get_diameter(rectangle):
    return (rectangle.get_width() + rectangle.get_height()) / 2


def check_silver_with_another(coin, gold_ls, two_ls, five_ls):
    if isinstance(gold_ls, list) and len(gold_ls) > 0:
        avr = sum([get_diameter(x) for x in gold_ls]) / (len(gold_ls))
        ratios = [R_1z_5, R_10_5, R_20_5]
    elif isinstance(two_ls, list) and len(two_ls) > 0:
        avr = sum([get_diameter(x) for x in two_ls]) / (len(two_ls))
        ratios = [R_1z_2z, R_10_2z, R_20_2z]
    elif isinstance(five_ls, list) and len(five_ls) > 0:
        avr = sum([get_diameter(x) for x in five_ls]) / (len(five_ls))
        ratios = [R_1z_2z, R_10_2z, R_20_2z]
    else:
        return None
    diameter = get_diameter(coin)
    diffs = [abs(diameter / avr - x['r']) for x in ratios]
    idx_min = diffs.index(min(diffs))
    return ratios[idx_min]['f']


def check_silver(coin_ls, gold_ls=None, two_ls=None, five_ls=None):
    # create matrix:
    #                   1 if ratio is bigger or equal to 1
    #                   else [0, 1] - ratio
    m = np.full((len(coin_ls), len(coin_ls)), 1.)
    dic = {}
    # for each two coins add note about ratio according description above
    for i1 in range(len(coin_ls)):
        for i2 in range(len(coin_ls)):
            if i1 == i2:
                continue
            # get diameter of rectangle which contains coin -> get coins average diameter
            coin1_diameter = get_diameter(coin_ls[i1])
            coin2_diameter = get_diameter(coin_ls[i2])
            m[i1][i2] = coin1_diameter / coin2_diameter if coin1_diameter / coin2_diameter < 1 else 1
    # print(m)
    # list of ratio for each set of coins
    r_ls = [R_10_1z, R_10_20, R_20_1z]
    # until we're able to find the best ratio for each coin
    while np.min(m) != 1:
        mins = []
        for r in r_ls:
            temp_m = abs(m - r['r'])
            # get minimal difference between expected and real ratio
            mins.append([np.min(temp_m), np.where(temp_m == np.min(temp_m))])
        # find overall minimal error (the most proper combination of coins)
        f_half = [min_i[0] for min_i in mins]
        m_min = min(f_half)
        idx = f_half.index(m_min)
        # if we have the same value multiple times:
        #                           for each of them mark coin
        for i in range(len(mins[idx][1][0])):
            x, y = mins[idx][1][0][i], mins[idx][1][1][i]
            # set as marked
            m[x, y] = 1
            if x not in dic:
                dic[x] = r_ls[idx]["f"]
            if y not in dic:
                dic[y] = r_ls[idx]['s']
    for i in range(len(coin_ls)):
        # if there're coins which are not marked
        if i not in dic:
            # if we have no information about 5 gr diameter - no option to determine
            #                                                 what this coin is
            if not gold_ls:
                dic[i] = "Unknown"
            # else try to find the closest matching with 5 gr ratio
            else:
                res = check_silver_with_another(coin_ls[i], gold_ls, two_ls, five_ls)
                dic[i] = res if res else "unknown"
    return dic


def main(level='easy'):
    for n, file in enumerate(os.listdir(f'examples/{level}')):
        if not any([extension in file.lower() for extension in ['jpg', 'jpeg', 'png']]):
            continue
        # if n != 4:
        #     continue
        # -- read image from input and make edge detection to find ellipse next
        im = im_resize(io.imread(f'examples/{level}/{file}', as_gray=True))

        # plt.imshow(im, cmap='gray')
        # plt.title("image")
        # plt.show()
        # time.sleep(1)

        # -- add contrast
        im = get_contrasted(im, 15)

        # im = gaussian(im, sigma=1.5)
        # im = median(im, disk(2))

        # plt.imshow(im, cmap='gray')
        # plt.title("contrasted")
        # plt.show()
        # time.sleep(1)
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

        objects = remove_small_objects(objects, 400)

        # plt.imshow(objects, cmap='gray')
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

        ls_silver = []
        ls_5gr = []
        ls_2 = []
        ls_5 = []
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
                        coin_center_color = "5 zł"
                        ls_5.append(rect)
                    else:
                        coin_center_color = "5 gr"
                        ls_5gr.append(rect)
                        # ls_silver.append((rect.get_height()+rect.get_width())/2)
                elif coin_center_color == "Silver":
                    two = check_if_two(im_rgb[minr:maxr, minc:maxc, :])
                    if two:
                        coin_center_color = "2 zł"
                        ls_2.append(rect)
                    else:
                        ls_silver.append(rect)
                        continue
                ax.add_patch(rect)
                ax.text(rect.get_x(), rect.get_y() - 3, coin_center_color, fontdict=font)
                ax.axis('off')
        if len(ls_silver) > 0:
            names = check_silver(ls_silver, ls_5gr, ls_2, ls_5)
            for i in range(len(ls_silver)):
                rect = ls_silver[i]
                ax.add_patch(rect)
                ax.text(rect.get_x(), rect.get_y() - 3, names[i], fontdict=font)
        # print(ls_silver)
        plt.show()
        # plt.hist([get_diameter(rect) for rect in ls_silver])
        # plt.show()


def mainloop():
    main('easy')
    # main("medium")
    # main("hard")


if __name__ == '__main__':
    mainloop()
