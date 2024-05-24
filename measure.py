import dlib
import cv2
import os
import numpy as np
from scipy.optimize import minimize
import argparse
from shapely.geometry import Polygon





# dlibの顔検出器と顔のランドマーク検出器を初期化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def distance_to_circle_sum_of_squares(params, points):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) - r
        return np.sum(distances ** 2)

def evaluate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔検出
    faces = detector(gray)

    
    # 顔のランドマークを描画
    for i, face in enumerate(faces):
        landmarks = predictor(gray, face)
        font_scale = min(image.shape[1], image.shape[0]) / 1000
        font_thickness = int(font_scale * 2)
        
        # 外側のランドマークポイントを取得
        points = []
        for n in list(range(0, 17)):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        for n in list(range(26, 16, -1)):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        
        # 多角形の面積を計算
        contour = np.array(points)
        area = cv2.contourArea(contour)
        print(f"Polygon Area: {area}")

        points = contour
        center = np.mean(points, axis=0)

        # 顔の角度を計算
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)

        face_angle = np.arctan2(right[1] - left[1], right[0] - left[0])
        S = []
        for point in points:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            if np.pi / 10 <= angle - face_angle <= np.pi * 9 / 10:
                S.append(point)
        
        S = np.array(S)

        initial_guess = [center[0], center[1], np.mean(np.linalg.norm(S - center, axis=1))]
        result = minimize(distance_to_circle_sum_of_squares, initial_guess, args=(S,))
        cx, cy, r = result.x
        print(f"Fitted Circle: center=({cx}, {cy}), radius={r}")


        # 円の面積を計算
        circle_area = np.pi * r ** 2
        print(f"Circle Area: {circle_area}")

    
        circle_contour = np.array([[cx + r * np.cos(theta), cy + r * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 100)])
        circle_contour = circle_contour.astype(np.float32)

        
        points_poly = Polygon(points)
        circle_poly = Polygon(circle_contour)

        intersection = points_poly.intersection(circle_poly)
        intersection_area = intersection.area



        marugao_ratio = intersection_area / area
        print(f"Marugao ratio: {marugao_ratio:.3f}")


        # lower_points = []

        # for n in list(range(2, 15)):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     lower_points.append((x, y))
        
        # lower_points = np.array(lower_points)


        
        # S_area = cv2.contourArea(lower_points)
        # S_perimeter = cv2.arcLength(lower_points, False)
        # # Sの最初の点、中心、最後の点を結んだときの角度
        # angle = np.abs(np.arctan2(lower_points[0][1] - cy, lower_points[0][0] - cx) - np.arctan2(lower_points[-1][1] - cy, lower_points[-1][0] - cx))
        
        # circularity = 2 * angle * S_area / S_perimeter ** 2

        # print(f"Circularity: {circularity:.3f}")
        
        
        # 多角形を描画
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=font_thickness*2)

        # 円を描画
        circle_contour = np.array(circle_contour, np.int32)
        circle_contour = circle_contour.reshape((-1, 1, 2))
        cv2.polylines(image, [circle_contour], isClosed=True, color=(0, 0, 255), thickness=font_thickness*2)

        
       
        # Roundness Ratioを円の上のテキストとして描画
        text = f"Marugao Degree: {marugao_ratio:.3f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        
        text_x = int(cx)
        text_y = int(cy - r - 10)
        
        # テキストが画像の右端を超えないように調整
        if text_x + text_size[0] > image.shape[1]:
            text_x = image.shape[1] - text_size[0] - 10

        # テキストが画像の上端を超えないように調整
        if text_y - text_size[1] < 0:
            text_y = text_size[1] + 10

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness*2)

        

    

    return image

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the image file or directory")
    parser.add_argument("--dir", "-d", type=str, help="Path to the output directory")

    args = parser.parse_args()

    # ファイルパスの設定
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    image_paths = [args.image]
    output_dir = args.dir


    # image_pathが画像かディレクトリかを判定(JPEG)
    if os.path.isdir(image_paths[0]):
        image_paths = [os.path.join(image_paths[0], file) for file in os.listdir(image_paths[0])]


    for image_path in image_paths:
        
        image = cv2.imread(image_path)

        evaluated = evaluate_image(image)
        image_name = os.path.basename(image_path)

        cv2.imwrite(f"{output_dir}/output_{image_name}", evaluated)

        



    
    


# # 画像の表示
# cv2.imshow("Landmarks", image)
# key = cv2.waitKey(0)

# # 'q'キーが押された場合のみウィンドウを閉じる例
# if key == ord('q'):
#     cv2.destroyAllWindows()