
# Imports:
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from IP.image_utils import *
from DL.TrainChanges import calculate_diff_per_image, get_crop_coordinates, get_extracted_img

filenames = ['good_data.csv', 'bad_data.csv']

def define_graph_configure(columns_amount):
    '''
    This function defines the configure of the graph
    :param columns_ammount: amount of columns
    :return: configure of graph
    '''
    if columns_amount == 1:
        return 1, 1
    return 2, 2

def avg(data):
    '''
    This function calculates average of a given data
    :param data: data
    :return: average of data
    '''
    return sum(data) / len(data)

def regression(column, data, title_text):
    '''
    This function calculates linear regression for a given data and column
    :param column: a column
    :param data: data
    :param title_text: file name in text mode to show as a title
    :return: average results of calculation
    '''

    _columns_titles = list(data.columns)  # get titles of columns
    _columns_titles.pop(0)  # pop image name title
    _x = [i for i in range(len(column))]

    _x = np.array(_x)
    _x = np.reshape(_x, (-1, 1))
    _last_index = len(column) - 1  # keep last index to start with in the prediction
    _new_x = [_last_index]
    i = 0
    j = 0

    _model = LinearRegression()  # define Linear Regression model

    _new_x = np.array(_new_x)
    _new_x = np.reshape(_new_x, (-1, 1))

    columns_amount = len(_columns_titles)
    _w, _h = define_graph_configure(columns_amount)

    fig, axs = plt.subplots(_h, _w, constrained_layout=True, squeeze=False, figsize=(4.2, 3))

    title = f"Filename:\n{title_text}"
    fig.suptitle(title, fontsize=10)

    avg_results = {}
    # predict data before regression
    for _column_title in _columns_titles:
        _column_values_t = data[_column_title].values[:]  # get column of values
        _column_values = np.array(_column_values_t)  # get column of values

        _new_result = -1

        try:
            _model.fit(_x, _column_values)  # fit model
            _y_pred = _model.predict(_x)
            _y_new = _model.predict(_new_x)

            x1 = column

            y1, y2 = _column_values_t.tolist()[:], _y_new.tolist()[:]
            _new_result = "%.2f" % y2[0]

            axs[i, j].plot(_x, y1, '-', _new_x[0], y2, 'o')
            avg_results.update({_column_title: y2[0]})
        except:
            pass

        axs[i, j].set_xlabel(f'{_column_title} ({_new_result})', fontsize=8)
        axs[i, j].set_ylabel('value', fontsize=8)

        j += 1
        if j == _w:
            j = 0
            i += 1

    return avg_results


def calculate_regression(filename):
    '''
    This function calculates linear regression for a given filename
    :param filename: filename
    :return: results of linear regression
    '''
    df = pd.read_csv(filename)  # get data frame

    columns_file = df.columns  # get titles of columns

    img_name_column = df[columns_file[0]].values.tolist()  # get column of image name
    results = regression(img_name_column, df, filename)
    return results


results_dict = {}
def keep_results_in_dict():
    '''
    This function calculates linear regression for each classified data
    :return: None
    '''
    for filename in filenames:
        print(filename)
        results = calculate_regression(filename)  # calculate regression, get results for top, bottom, left and right
        print(results)
        print('================================================')

        results_avg = sum(results.values()) / 4
        results.update({'AVG': results_avg})  # keep average of results
        results_dict.update({filename: results})


def choose_side_classification(change_img_results, side, index):
    '''
    This function chooses a classification for a given side
    :param change_img_results: results of changes in image
    :param side: current side in image
    :param index: index of side
    :return: 1 - classified as a good change, -1 - classifies as a bad change
    '''
    good = abs(results_dict[filenames[0]][side] - change_img_results[index])
    bad = abs(results_dict[filenames[1]][side] - change_img_results[index])

    if good > bad:
        return 1
    else:
        return -1


def choose_classification(sides_classifier):
    '''
    This function chooses final classification for image
    :param sides_classifier: a lists of sides classifications
    :return: 1 - classified as a good changed image, -1 - classifies as a bad changed image
    '''
    if sum(sides_classifier) > 0:
        return 1
    return -1

def classify_image(img_name, source_img, change_img):
    '''
    This function classify an image
    :param img_name: image name
    :param source_img: source image
    :param change_img: changed image
    :return: classification
    '''
    # Get crop coordinates
    crop = get_crop_coordinates(img_name)
    change_img_results = calculate_diff_per_image(source_img, change_img, crop)

    # add average of changes
    change_avg = sum(change_img_results) / 4
    change_img_results += (change_avg,)

    # classify sides of changes in image
    right = choose_side_classification(change_img_results, 'Right', 0)
    left = choose_side_classification(change_img_results, 'Left', 1)
    top = choose_side_classification(change_img_results, 'Top', 2)
    bottom = choose_side_classification(change_img_results, 'Bottom', 3)
    avg = choose_side_classification(change_img_results, 'AVG', 4)

    classified = choose_classification([right, left, top, bottom, avg])  # classify image
    return classified


def classification(path):
    '''
    This function classifies each image in a given directory
    :param path: a path to a directory
    :return: a dictionary of classified images in the given path
    '''
    changes_dict = {}
    for image in glob.glob(f"{path}\*{IMG_SUFFIX}"):
        img_name, source_img, change_img = get_extracted_img(image)

        try:
            classified = classify_image(img_name, source_img, change_img)
            changes_dict.update({img_name: classified})

        except Exception as e:
            print(e)
    return changes_dict
