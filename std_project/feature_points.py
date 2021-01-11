import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

class FeaturePointsController():
    def __init__(self, input_img, blur_thresh=3):
        self._input_img_path = input_img
        self._blur_thresh = blur_thresh
        self._img = self.read_img(self._input_img_path)

    def read_img(self, img_path):
        return cv2.imread(img_path)

    def show_img(self, img, title):
        cv2.imshow(title, img)

    def find_feature_points(self, input_img, blur_thresh=3):

        blur_img = cv2.GaussianBlur(input_img,(blur_thresh, blur_thresh),
                                    cv2.BORDER_DEFAULT)
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(grey_img, 70, 70)
        self.show_img(canny_img, 'canny img')

        _, bin_img = cv2.threshold(canny_img, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    def get_segmentation_img(self, img, thresh=150):
        _, bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        return bin

    def draw_rectangles(self, img, rects, color):
        for (xA, yA, xB, yB) in rects:
            cv2.rectangle(img, (xA, yA), (xB, yB), color, 1)

    def _remove_overlap_rectangles(self, rects, thresh=0.5):
        return non_max_suppression(rects, probs=None, overlapThresh=thresh)

    def _is_proper_size_rectangle(self, w, h):
        return w > 5 and h > 10 and w < 100 and h < 300

    def known_objects_overlap(self):
        pass

    def compare_to_another_threshold(self):
        pass

    def check_edges_variance(self):
        pass

    def run(self):
        rects = self.find_feature_points(self._img)
        self.draw_rectangles(self._img, rects, (0, 255, 0))
        self.show_img(self._img, 'feature points')
        blur_img = cv2.GaussianBlur(self._img, (3, 3), cv2.BORDER_DEFAULT)
        grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        bin_img = self.get_segmentation_img(grey_img)
        self.show_img(bin_img, 'segmentation')

        cv2.waitKey()
        cv2.destroyAllWindows()
