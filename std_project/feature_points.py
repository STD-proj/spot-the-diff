import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import image_utils

class FeaturePointsController():

    def __init__(self, input_img, blur_thresh=3):
        # FIXME after the class will be well defined
        self._input_img_path = input_img
        self._blur_thresh = blur_thresh
        self._img = image_utils.read_img(self._input_img_path)

    def find_feature_points(self, input_img, blur_thresh=3):
        # perform gaussianBlur to remove irrelevant noise
        blur_img = cv2.GaussianBlur(input_img,(blur_thresh, blur_thresh),
                                    cv2.BORDER_DEFAULT)
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        # detect edges with Canny
        canny_img = cv2.Canny(grey_img, 70, 70)
        image_utils.show_img(canny_img, 'canny img')

        _, bin_img = cv2.threshold(canny_img, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter the contours by size and overlaps
        return self.first_pass_filtering(contours)

    def first_pass_filtering(self, contours):
        rects = []
        # filter by rectangle size
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if self._is_proper_size_rectangle(w, h):
                rects.append((x, y, x + w, y + h))

        # remove overlap rectangles
        rects = self._remove_overlap_rectangles(np.array(rects))
        return rects

    def second_pass_filtering(self, orig_img, bin_img, rects):
        new_rects = []
        # compare each rectangle to the its segmentation image
        for index, (xA, yA, xB, yB) in enumerate(rects):
            crop_img_bin = bin_img[yA:yB, xA:xB]
            white_count = cv2.countNonZero(crop_img_bin)
            black_count = crop_img_bin.size - white_count
            # calculate the percentage of each color
            white_per = white_count / crop_img_bin.size
            black_per = black_count / crop_img_bin.size

            if white_per > 0.85 or black_per > 0.85:
                continue
            new_rects.append((xA, yA, xB, yB))

            if index > 0: continue # TODO: remove later
            sums = []
            # copy the segment to the original image
            for y in range(crop_img_bin.shape[0]):
                for x in range(crop_img_bin.shape[1]):
                    if crop_img_bin[y, x] != 0:
                        sums.append(orig_img[y+yA, x+xA])

            avg_a = sum([color[0] for color in sums]) / len(sums)
            avg_b = sum([color[1] for color in sums]) / len(sums)
            avg_c = sum([color[2] for color in sums]) / len(sums)

            for y in range(crop_img_bin.shape[0]):
                for x in range(crop_img_bin.shape[1]):
                    if crop_img_bin[y, x] == 0:
                        orig_img[y+yA, x+xA] = [avg_a, avg_b, avg_c]


        image_utils.show_img(orig_img, 'orig')
        return new_rects

    def get_segmentation_img(self, img, thresh=160):
        _, bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        return bin

    def draw_rectangles(self, img, rects, color):
        for (xA, yA, xB, yB) in rects:
            cv2.rectangle(img, (xA, yA), (xB, yB), color, 1)

    def _remove_overlap_rectangles(self, rects, thresh=0.5):
        return non_max_suppression(rects, probs=None, overlapThresh=thresh)

    def _is_proper_size_rectangle(self, w, h):
        return 5 < w < 100 and 10 < h < 100

    def known_objects_overlap(self):
        pass

    def compare_to_another_threshold(self):
        pass

    def check_edges_variance(self):
        pass

    def choose_changes(self, rects):
        pass

    def run(self):
        copy_img = image_utils.read_img(self._input_img_path)
        image_utils.show_img(copy_img, 'first')
        copy_img2 = image_utils.read_img(self._input_img_path)
        copy_img3 = image_utils.read_img(self._input_img_path)
        rects = self.find_feature_points(copy_img)

        self.draw_rectangles(copy_img, rects, (0, 255, 0))
        image_utils.show_img(copy_img, 'feature points')
        blur_img = cv2.GaussianBlur(copy_img2, (3, 3), cv2.BORDER_DEFAULT)
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        bin_img = self.get_segmentation_img(grey_img)
        rects = sorted(rects, key=lambda tup: tup[0])
        rects = self.second_pass_filtering(self._img, bin_img, rects)

        image_utils.show_img(bin_img, 'segmentation')
        self.draw_rectangles(copy_img3, rects, (0, 0, 255))
        image_utils.show_img(copy_img3, 'second')

        cv2.waitKey()
        cv2.destroyAllWindows()
