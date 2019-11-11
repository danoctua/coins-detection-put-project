import sys

import numpy as np
from matplotlib import pylab as plt
from scipy.signal import medfilt
from skimage import io, data, color, img_as_ubyte
from skimage.transform import resize
from skimage.filters import threshold_yen, gaussian, median
from skimage.feature import canny
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, binary_opening, remove_small_holes, dilation, disk
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage.draw import ellipse_perimeter, circle_perimeter


def im_resize(img):
    return resize(img, (250, 250))


def main():
    for n in range(1, 4):
        # read image from input and make edge detection to find ellipse next
        im = im_resize(io.imread(f'examples/easy/coin{n}.jpg', as_gray=True))
        im = gaussian(im, sigma=2)
        thresh = threshold_yen(im)
        binary = im > thresh
        binary_cleaned = remove_small_holes(binary, 1000)
        binary_cleaned = remove_small_objects(binary_cleaned, 1000)

        binary_smooth = medfilt(binary_cleaned, 3)
        binary_smooth = dilation(canny(binary_smooth))

        # show detected edges for de
        plt.imshow(1-binary_smooth, cmap='binary')
        plt.show()

        hough_radii = np.arange(15, 50, 2)
        hough_res = hough_circle(binary_smooth, hough_radii)

        # Select the most prominent 3 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3)
        print(hough_res, hough_res.shape)


        im_rgb = im_resize(io.imread(f'examples/easy/coin{n}.jpg'))
        if len(hough_res) > 0:
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=im_rgb.shape)
                im_rgb[circy, circx] = (1, 0, 0)

            # best = list(result[-1])
            # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
            # orientation = best[5]
            #
            # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            plt.imshow(im_rgb)

            plt.show()
        else:
            print("no circle found")


if __name__ == '__main__':
    main()