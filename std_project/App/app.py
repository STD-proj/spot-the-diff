import cv2
import os
import create_mask as cm
import HiFill.repainting as re
import glob
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# image = 'samples/testset/1.jpg'
# fp = cm.FeaturePointsController(image)
# fp.run()
# mask = 'samples/maskset/1.jpg'
# re.run(image, mask)
# result = glob.glob(f"results/*.jpg")[0]

if __name__ == "__main__":

    # Using flask
    app = Flask(__name__)
    CORS(app)
    app.secret_key = 's3cr3t'

    @app.route('/process', methods=['POST'])
    def upload_file():
        file = request.form['img']
        img = 'samples/testset/' + os.path.basename(file)
        fp = cm.FeaturePointsController(img)
        fp.run()
        mask = 'samples/maskset/' + os.path.basename(file)
        re.run(img, mask)
        result = f"results/{os.path.basename(file)[:-4]}_inpainted.jpg"
        return result , 200


    app.run("localhost", "8000", debug=True)
