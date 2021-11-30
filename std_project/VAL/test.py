import unittest
from PIL import Image 

class TestProject(unittest.TestCase):

    # test that we made by running the code, and we document it.

    def test_feature_points(self, image):
        """
        Get an image and see if it find feature points in a randomal image
        """
    def test_picture_classification(self):
        """
        tetst the image classification by the algorithm
        """
        
    def test_function_argument(self, input, type):
        """
        Test argument of functions
        """
        if(type == 'image'):
            try:
                im = Image.open(input)
            except IOError:
                print("not an image type,please insert an image")
                exit()
      #  if not isinstance(input, type):
      #      print('argument must be {} type'.format(type))
      #      exit()
        

