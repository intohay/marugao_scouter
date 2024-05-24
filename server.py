from flask import Flask, request, send_file
import io
from measure import evaluate_image
import cv2
import numpy as np
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    # 画像の読み込み
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)


    processed_img = evaluate_image(img)

    # 加工後の画像をバッファに保存
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    img_io = io.BytesIO(img_encoded)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
