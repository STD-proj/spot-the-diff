import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from IP.feature_points import *

'''
For each changed image, check the diff between in and the source.
Then pass on the neighbors of the changed pixels and check the diff.
Calculate probability of change. 
'''

IMG_SUFFIX = ".jpg"
path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(path, "../Data/examples")
change_path = os.path.join(path, "../Data")
good_changes = "good_changes"
bad_changes = "bad_changes"


def calculate_diff(source_img, change_img, crop):
    sums_change = 0
    sums_source = 0
    xA, yA, xB, yB = crop

    # Check difference between pixels on x axis
    for pixel in range(xB-xA):
        sums_change += abs(change_img[yA][pixel + xA] - change_img[yA-1][pixel + xA])
        sums_source += abs(source_img[yA][pixel + xA] - source_img[yA-1][pixel + xA])
        sums_change += abs(change_img[yA][pixel + xA] - change_img[yA+1][pixel + xA])
        sums_source += abs(source_img[yA][pixel + xA] - source_img[yA+1][pixel + xA])

    # Check difference between pixels on y axis
    for pixel in range(yB-yA):
        sums_change += abs(change_img[pixel + yA][xA] - change_img[pixel + yA][xA-1])
        sums_source += abs(source_img[pixel + yA][xA] - source_img[pixel + yA][xA-1])
        sums_change += abs(change_img[pixel + yA][xA] - change_img[pixel + yA][xA+1])
        sums_source += abs(source_img[pixel + yA][xA] - source_img[pixel + yA][xA+1])

    average = sum(abs(sums_source - sums_change))/3
    return average

def calculate_diff_per_image(source_img, change_img, crop):
    '''
    This function calculates difference per image
    :param source_img: source image
    :param change_img: change image
    :param crop: crop
    :return: difference in right, left, top and bottom sides
    '''
    xA, yA, xB, yB = crop

    right_sum_source = 0
    left_sum_source = 0
    top_sum_source = 0
    bottom_sum_source = 0

    right_sum_change = 0
    left_sum_change = 0
    top_sum_change = 0
    bottom_sum_change = 0

    y, x, _ = source_img.shape

    print(source_img.shape)
    print(crop)

    # Check difference between pixels on x axis
    for pixel in range(xB-xA):
        if yB+1 >= y: yB = y-2
        top_sum_change += abs(change_img[yA][pixel + xA] - change_img[yA-1][pixel + xA])
        bottom_sum_change += abs(change_img[yB][pixel + xA] - change_img[yB+1][pixel + xA])

        top_sum_source += abs(source_img[yA][pixel + xA] - source_img[yA-1][pixel + xA])
        bottom_sum_source += abs(source_img[yB][pixel + xA] - source_img[yB+1][pixel + xA])



    # Check difference between pixels on y axis
    for pixel in range(yB-yA):
        if xB +1 >= x: xB = x-2
        left_sum_change += abs(change_img[pixel + yA][xA] - change_img[pixel + yA][xA-1])
        right_sum_change += abs(change_img[pixel + yA][xB] - change_img[pixel + yA][xB+1])

        left_sum_source += abs(source_img[pixel + yA][xA] - source_img[pixel + yA][xA-1])
        right_sum_source += abs(source_img[pixel + yA][xB] - source_img[pixel + yA][xB+1])

    right, left, top, bottom = sum(abs(right_sum_change - right_sum_source)), sum(abs(left_sum_change - left_sum_source)), \
                               sum(abs(top_sum_change - top_sum_source)), sum(abs(bottom_sum_change - bottom_sum_source))

    return right, left, top, bottom


def create_data(img_name, absolute_data):
    current_row = [img_name]
    current_row += [data for data in absolute_data]
    return current_row


def get_extracted_img(image):
    # Calculate source image path
    source_sub_name = image.split("\\")[-1]
    source_sub_name = source_sub_name.split("/")[-1].split(".")[0]
    img_name = source_sub_name

    source_sub_name = source_sub_name.split("_")[:1]
    source_img_name = os.path.join(source_path, "_".join(source_sub_name) + IMG_SUFFIX)

    source_img = read_img(source_img_name)
    change_img = read_img(image)

    return img_name, source_img, change_img


def run(path):
    average = 0
    avg_crop = 0
    count = 0
    change_img = None

    data = []
    img_name = ''

    for image in glob.glob(f"{path}\*{IMG_SUFFIX}"):
        # Calculate source image path
        source_sub_name = image.split("\\")[-1]
        source_sub_name = source_sub_name.split("/")[-1].split(".")[0]
        img_name = source_sub_name

        source_sub_name = source_sub_name.split("_")[:1]
        source_img_name = os.path.join(source_path, "_".join(source_sub_name) + IMG_SUFFIX)

        source_img = read_img(source_img_name)
        change_img = read_img(image)

        print(img_name)

        try:
            # Get crop coordinates
            crop = get_crop_coordinates(img_name)
            average += calculate_diff(source_img, change_img, crop)
            current_data = calculate_diff_per_image(source_img, change_img, crop)
            data.append(create_data(img_name, current_data))

            avg_crop += avg_image_change(change_img, crop)
            count += 1
        except:
            print("failed to save data")
            return

    print("Final average: ", average / count)
    print("Final average crop: ", avg_crop / count)
    print('\n========================================\n')
    return change_img, img_name, data

def get_crop_coordinates(img_name):
    xA, yA, xB, yB = [int(index) for index in img_name.split("_")[2:]]
    return xA, yA, xB, yB

def surface_of_crop(xA, yA, xB, yB):
    return (xB - xA) * (yB - yA)

def avg_image_change(change_img, crop):
    xA, yA, xB, yB = crop
    if yA == 0: yA += 1
    if xA == 0: xA += 1
    sur_in = surface_of_crop(xA, yA, xB, yB)
    in_crop = sum(sum(sum(change_img[yA:yB, xA:xB])))/3
    out_crop = sum(sum(sum(change_img[yA-1:yB+1, xA-1:xB+1])))/3
    print(in_crop, end=', ')
    print(out_crop)

    return abs(out_crop - in_crop)


def main():
    path = os.path.join(change_path, bad_changes)
    print(path)
    change_img, img_name, data = run(path)

    filename = 'bad_data.csv'
    bad_data = pd.DataFrame(data, columns=['Img_name', 'Right', 'Left', 'Top', 'Bottom'])
    bad_data.to_csv(filename, index=False)

    path = os.path.join(change_path, good_changes)
    print(path)
    change_img, img_name, data = run(path)
    filename = 'good_data.csv'
    good_data = pd.DataFrame(data, columns=['Img_name', 'Right', 'Left', 'Top', 'Bottom'])
    good_data.to_csv(filename, index=False)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
