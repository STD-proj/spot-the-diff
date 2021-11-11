
# -- Imports --
import argparse
from IP.feature_points import *
from DL.Regression import *
# from DL.Regression import classify_image, get_extracted_img


def main(image):
    '''
    This is the main function.
    In this function, all the process for a given image is carried out.
    :param image: a given image from user.
    :return: None
    '''

    counter = 0
    delete_all_files_in_folder()  # delete all previous changed images

    keep_results_in_dict()  # calculate linear regression for each classified data and keep results in a dictionary

    while counter < AMOUNT_OF_RUNS:  # bound amount of tries
        fp_controller = FeaturePointsController(image)
        fp_controller.run()  # run a function to create a change in an image
        img = get_changed_image(DIRECTORY)  # get changed image


        img_name, source_img, change_img = get_extracted_img(img)  # extract the changed image into filename, source image and current image
        classified = classify_image(img_name, source_img, change_img)  # classify changed image
        print(img_name, classified)

        if classified == 1:  # if succeed to create a good changed image - break loop
            print(f'Image {img_name} succeeded to classify.')
            break
        else:
            print(f'Image {img_name} could not be classified.')
        os.remove(f'{img}')  # remove changed image
        counter += 1
    else:  # if could not find a good change
        print(f'Could not find any good change for image {image}.')
    return


# When user runs the program
if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="path to input image")
        args = vars(ap.parse_args())
        image = args["image"]
        main(image)  # run program
        plt.show()  # show linear regression on graph
    except Exception as e:
        print(f'Can not run program due to the next reason: {e}')
        exit()
