import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, render_template, send_from_directory
import eye_cropper
import tensorflow as tf

COUNT = 0
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global COUNT
    if request.method == 'POST':
        if request.files:
            img = request.files['image']
            img.save('upload/{}.jpg'.format(COUNT))
            #file.save(os.path.join(app.config["IMAGE_UPLOADS"], file.filename))

        # Process uploaded image
        L_eye, R_eye = eye_cropper.process_face_img('upload/{}.jpg'.format(COUNT))

        # Load in saved model
        saved_model = tf.keras.models.load_model('best_model_2.h5')
        L_eye_array = np.asarray(L_eye)
        L_eye_array = L_eye_array.reshape(1, 80, 80, 3)

        R_eye_array = np.asarray(R_eye)
        R_eye_array = R_eye_array.reshape(1, 80, 80, 3)

        left_eye_pred = saved_model.predict(L_eye_array)[0][0].round()
        right_eye_pred = saved_model.predict(L_eye_array)[0][0].round()

        final_preds = eye_cropper.convert_words([left_eye_pred, right_eye_pred])

        eye_cropper.save_cropped_eyes(
            '{}'.format(COUNT), L_eye, R_eye, final_preds[0], final_preds[1])

        files = eye_cropper.get_file_logs('upload')
        COUNT += 1

    return render_template('submit.html', data=[final_preds[0], files[0]])

@app.route('/load_img/<img_name>')
def load_img(img_name):
    global COUNT

    return send_from_directory('upload', img_name)

if __name__ == '__main__':
    app.run(debug=True)
