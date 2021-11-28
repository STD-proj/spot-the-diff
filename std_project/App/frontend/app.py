from __future__ import print_function
from flask import Flask, request, jsonify
from flask_cors import CORS

import os

app = Flask(__name__)
CORS(app)
app.secret_key = 's3cr3t'
app.debug = True
app._static_folder = os.path.abspath("templates/static/")

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    img = request.form['img']
    print("img=", img)
    return jsonify("res back")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)