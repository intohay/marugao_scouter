from flask import Flask, request, send_file
from PIL import Image
import io
from measure import evaluate_image

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    # 画像の読み込み
    img = Image.open(file.stream)


    processed_img = evaluate_image(img)

    # 加工後の画像をバッファに保存
    img_io = io.BytesIO()
    processed_img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
