import cv2

def read_img(img_path):
    return cv2.imread(img_path)

def show_img(img, title):
    cv2.imshow(title, img)
