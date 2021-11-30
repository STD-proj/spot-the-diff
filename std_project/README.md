# Spot the Difference

### Objective
The project goal is to develop the ability to create a new image with good change from the given user's image. 

### Implementation
The project was developed in 2 main phases:
+ Image Processing - creating tool for modifying images, this ability used for creating the data-set and for the actual game flow.
+ Deep Learning - classifying a given image with randomized change as a 'good enough' or 'bad' change.

### How to
+ run the feature points algorithm: ``python feature_points.py -i <path/to/image> -n <num_of_changes>`` inside the IP folder
+ create the dataset: ``python create_data.py`` inside the Data folder
+ run regression: ``python main.py -i <path/to/image>``
+ use the website: ``python app.py`` inside the App folder and open the html file
