import base64
import io

from flask import Flask, render_template, request
from gad import highlightFace
import cv2
import argparse
import numpy as np
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def after():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')

    args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    padding = 20
    #img = request.files['file1']
    #img.save(os.path.join(app.root_path, 'static/file.jpg'))
    #img_by = io.BytesIO()
    #img.save(img_by, "JPEG")
    #encoded_img_data = base64.b64encode(img_by.getvalue())
    #input_image = cv2.imread(os.path.join(app.root_path, 'static/file.jpg'))
    #input_image = cv2.imread(encoded_img_data.decode('utf-8'))

    input_image = cv2.imdecode(np.fromstring(request.files['file1'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    resultImg, faceBoxes = highlightFace(faceNet, input_image)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = input_image[max(0, faceBox[1] - padding):
                           min(faceBox[3] + padding, input_image.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                                :min(faceBox[2] + padding,
                                                                                     input_image.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    # result_img_by = io.BytesIO()
    #resultImg.save(result_img_by, "JPEG")
    #encoded_result_img = base64.b64encode(result_img_by.getvalue())
    display_gender = f'Gender: {gender}'
    display_age = f'Age: {age[1:-1]} years'

    return render_template("index.html", gender=display_gender, age=display_age)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = int(os.environ.get('PORT', 8080)))