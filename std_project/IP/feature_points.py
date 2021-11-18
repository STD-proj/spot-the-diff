import argparse
from msilib import Directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from IP.image_utils import *
import random

class FeaturePointsController():
    '''
    This class controls the features of the points.
    '''
    def __init__(self, input_img, blur_thresh=3):
        '''
        This is the constructor of the class
        :param input_img: an image
        :param blur_thresh: threshold, default sets to 3
        '''
        # FIXME after the class will be well defined
        self._input_img_path = input_img
        self._blur_thresh = blur_thresh
        self._img = read_img(self._input_img_path)
        self._filename = input_img.split('\\')[-1]
        self._filename = self._filename.split('/')[-1]
        self._filename = self._filename[:self._filename.find('.')]

    def find_feature_points(self, input_img, blur_thresh=3):
        '''
        This function finds features in image.
        :param input_img: an iamge
        :param blur_thresh:  threshold, default sets to 3
        :return: filtered contours
        '''
        blur_img = cv2.GaussianBlur(input_img,(blur_thresh, blur_thresh),
                                    cv2.BORDER_DEFAULT)  # perform gaussianBlur to remove irrelevant noise
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

        canny_img = cv2.Canny(grey_img, 128, 128)  # detect edges with Canny

        _, bin_img = cv2.threshold(canny_img, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        return self.first_pass_filtering(contours)   # filter the contours by size and overlaps

    def first_pass_filtering(self, contours):
        '''
        This function filters a first pass
        :param contours: contours
        :return: chosen rectangles
        '''
        rects = []
        # filter by rectangle size
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if self._is_proper_size_rectangle(w, h):
                rects.append((x, y, x + w, y + h))

        rects = self._remove_overlap_rectangles(np.array(rects))  # remove overlap rectangles
        return rects

    def surface_of_crop(self, xA, yA, xB, yB):
        '''
        This function finds surface of a crop depends on two points
        :param xA: top left x value
        :param yA: top left y value
        :param xB: bottom right x value
        :param yB: bottom right y value
        :return: surface
        '''
        return (xB - xA) * (yB - yA)

    def second_pass_filtering(self, bin_img, rects):
        '''
        This function filters a second pass
        :param bin_img: binary image
        :param rects: rectangles
        :return: new rectangles and new changes
        '''
        # calculate surface of each crop to remove small areas
        surfaces = [self.surface_of_crop(xA, yA, xB, yB) for (xA, yA, xB, yB) in rects]
        avg_surface = sum(surfaces) / len(surfaces)

        # calculate erotion of binary image to remove small noises in pic
        kernel = np.ones((5, 5), 'uint8')
        bin_img = cv2.erode(bin_img, kernel, iterations=1)

        new_rects = []

        # compare each rectangle to its segmentation image
        for index, (xA, yA, xB, yB) in enumerate(rects):
            if self.surface_of_crop(xA, yA, xB, yB) < avg_surface:  # if current crop is too small - skip current loop
                continue
            crop_img_bin = bin_img[yA:yB, xA:xB]

            white_count = cv2.countNonZero(crop_img_bin)
            black_count = crop_img_bin.size - white_count

            # calculate the percentage of each color
            white_per = white_count / crop_img_bin.size
            black_per = black_count / crop_img_bin.size

            if white_per > 0.85 or black_per > 0.85:  # if current crop has too much percentage of any color - skip current loop
                continue
            new_rects.append((xA, yA, xB, yB))  # keep current rectangle

        return new_rects

    def apply_changes(self, bin_img, rects, num_changes=1):

        source_img = read_img(self._input_img_path)
        # calculate erotion of binary image to remove small noises in pic
        kernel = np.ones((5, 5), 'uint8')
        bin_img = cv2.erode(bin_img, kernel, iterations=1)

        if num_changes > len(rects):
            num_changes = len(rects)
        changes = random.sample(rects, num_changes) #randomize few changes
        change_img = read_img(self._input_img_path)
        idxs = []

        for index, (xA, yA, xB, yB) in enumerate(changes):
            crop_img_bin = bin_img[yA:yB, xA:xB]

            white_count = cv2.countNonZero(crop_img_bin)
            black_count = crop_img_bin.size - white_count

            # calculate the percentage of each color
            white_per = white_count / crop_img_bin.size
            black_per = black_count / crop_img_bin.size

            if white_per <= black_per:
                change_img = self.remove_color(source_img, crop_img_bin, xA, yA, is_black=False)
            else:
                change_img = self.remove_color(source_img, crop_img_bin, xA, yA)
            idxs.append((rects.index((xA, yA, xB, yB)), (xA, yA, xB, yB)))

        return change_img, self._filename, idxs

    def remove_color(self, orig_img, crop_img_bin, xA, yA, is_black=True):
        '''
        This function removes black or white area in image.
        :param orig_img: original image
        :param crop_img_bin: cropped binary image
        :param xA: top left x value
        :param yA: top left y value
        :param is_black: which color to remove
        :return: original image after the change
        '''
        sums = []
        # copy the segment to the original image
        for y in range(crop_img_bin.shape[0]):
            for x in range(crop_img_bin.shape[1]):
                if (is_black and crop_img_bin[y, x] == 0) or (not is_black and crop_img_bin[y, x] != 0):
                    sums.append(orig_img[y + yA, x + xA])

        avg_a = sum([color[0] for color in sums]) / len(sums)
        avg_b = sum([color[1] for color in sums]) / len(sums)
        avg_c = sum([color[2] for color in sums]) / len(sums)

        for y in range(crop_img_bin.shape[0]):
            for x in range(crop_img_bin.shape[1]):
                if (is_black and crop_img_bin[y, x] != 0) or (not is_black and crop_img_bin[y, x] == 0):
                    orig_img[y + yA, x + xA] = [avg_a, avg_b, avg_c]
        return orig_img

    def get_segmentation_img(self, img, thresh=160):
        '''
        This function returns the segmentation image
        :param img: an image
        :param thresh: threshold, default sets as 160
        :return: binary segmentation
        '''
        _, bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        return bin

    def draw_rectangles(self, img, rects, color):
        '''
        This function draws rectangles on a given image
        :param img: a given image
        :param rects: a list of rectangles
        :param color: color
        :return: None
        '''
        for (xA, yA, xB, yB) in rects:
            cv2.rectangle(img, (xA, yA), (xB, yB), color, 1)

    def _remove_overlap_rectangles(self, rects, thresh=0.5):
        '''
        This function removes overlap rectangles
        :param rects: rectangles
        :param thresh: threshold
        :return: a list of rectangles
        '''
        return non_max_suppression(rects, probs=None, overlapThresh=thresh)

    def _is_proper_size_rectangle(self, w, h):
        '''
        This function checks if a given width and hright of rectangle is proper
        :param w: a width of rectangle
        :param h: a height of rectangle
        :return: is proper or not
        '''
        return 5 < w < 100 and 10 < h < 100

    def run(self, num_changes):
        '''
        This function runs all the program in current file
        :return: None
        '''

        # read image and keep copies of it
        feature_points_img = read_img(self._input_img_path)
        blur_img = read_img(self._input_img_path)
        final_rects_img = read_img(self._input_img_path)

        rects = self.find_feature_points(feature_points_img)  # find features in image
        self.draw_rectangles(feature_points_img, rects, (0, 255, 0))  # draw rectangles on image

        blur_img = cv2.GaussianBlur(blur_img, (3, 3), cv2.BORDER_DEFAULT)
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        bin_img = self.get_segmentation_img(grey_img)
        rects = sorted(rects, key=lambda tup: tup[0])
        rects = self.second_pass_filtering(bin_img, rects)
        self.draw_rectangles(final_rects_img, rects, (0, 0, 255))  # draw rectangles on image

        change_img, filename, idxs = self.apply_changes(bin_img, rects, num_changes)
        return change_img, idxs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-n", "--num_changes", required=True,
                    help="number of changes to apply")
    args = vars(ap.parse_args())
    image = args["image"]
    num_changes = args["num_changes"]

    fpc = FeaturePointsController(image)
    change_img, idxs = fpc.run(int(num_changes))

    show_img(fpc._img, "first img")
    show_img(change_img, "change img")
    fpc.draw_rectangles(change_img, [rect[1] for rect in idxs], (255, 0, 0))
    show_img(change_img, "change img with rects")
    cv2.waitKey()
    cv2.destroyAllWindows()

