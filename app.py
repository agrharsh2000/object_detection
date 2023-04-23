from flask import Flask, request, jsonify, render_template
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
       
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box, label, c_score = cv.detect_common_objects(image, confidence=0.01, model='yolov3-tiny')
        output = draw_bbox(image, box, label, c_score)
        num_objects = len(label)
        object_labels = set(label) 
        result = {
            'num_objects': num_objects,
            'object_labels': list(object_labels)
        }
        result_text = ', '.join(list(object_labels))
        return render_template('result.html', result=result_text)

    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
