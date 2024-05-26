import dlib
import cv2
import os
import numpy as np
from scipy.optimize import minimize
import argparse
from shapely.geometry import Polygon



predictor_path = "shape_predictor_68_face_landmarks.dat"

# dlibの顔検出器と顔のランドマーク検出器を初期化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def distance_to_circle_sum_of_squares(params, points):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) - r
        return np.sum(distances ** 2)

def get_world_points(shape, image, points):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = 1.0 * image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1)) # 歪み係数（無視）

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    print("rotation_matrix")
    print(rotation_matrix)
    print("translation_vector")
    print(translation_vector)
    
    inv_r_matrix = np.linalg.inv(rotation_matrix)

    print("inv_r_matrix")
    print(inv_r_matrix)


    z_world = -140

    world_coordinates = []


    print(image.shape)
    print(center[0], center[1])

    for point in points:
        
        image_point = np.array([point[0], point[1], 1.0], dtype="double")
        normalized_camera_point = np.linalg.inv(camera_matrix) @ image_point
        

        inv_r_t = inv_r_matrix @ translation_vector
        inv_r_n = inv_r_matrix @ normalized_camera_point

        scale = (z_world + inv_r_t[2]) / inv_r_n[2]

       
        camera_point = scale * normalized_camera_point
        
        world_point = inv_r_matrix @ (camera_point - translation_vector.flatten())

    
        world_coordinates.append((world_point[0], world_point[1]))
        
    
    


    return world_coordinates, rotation_matrix, translation_vector, camera_matrix, z_world

def calculate_angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product)
    return angle


def make_positive(points):
    min_x = points[0][0]
    min_y = points[0][1]

    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        
        if point[1] < min_y:
            min_y = point[1]

    positive_points = []
    for point in points:
        positive_point = (point[0] - min_x + 1, point[1] - min_y + 1)
        positive_points.append(positive_point)

    return positive_points, min_x, min_y

        
        
def get_outline(landmarks):
    # 外側のランドマークポイントを取得
    points = []
    for n in list(range(0, 17)):
        points.append(landmarks[n])
        # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    for n in list(range(26, 16, -1)):
        
        points.append(landmarks[n])
        # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return points

def convert_world_to_image(r_matrix, t_vector, camera_matrix, z_point, points):
    converted_points = []
    for point in points:
        camera_point = r_matrix @ np.array([point[0], point[1], z_point]) + t_vector.flatten()
        
        normalized_camera_point = camera_point / camera_point[2]

        converted_point = camera_matrix @ normalized_camera_point

        converted_points.append(converted_point[:2])

    return converted_points



def evaluate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔検出
    faces = detector(gray)

    
    # 顔のランドマークを描画
    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        original_landmarks = np.array([(p.x, p.y) for p in shape.parts()])


        landmarks, r_matrix, t_vector, camera_matrix, z_point = get_world_points(shape, image, original_landmarks)
        

        font_scale = min(image.shape[1], image.shape[0]) / 1000
        font_thickness = int(font_scale * 2)
        
        points = get_outline(landmarks)
        

        # すべての点を正にする
        print(points)
        points, offset_x, offset_y = make_positive(points)
        for i in range(len(points)):
            points[i] = (points[i][0].astype(np.float32), points[i][1].astype(np.float32))

        

        print(points)
        # 多角形の面積を計算
        print(points[0][0].dtype)
        contour = np.array(points)
        area = cv2.contourArea(contour)
        print(f"Polygon Area: {area}")

        points = contour
        center = np.mean(points, axis=0)

        # 顔の角度を計算
        left = (landmarks[39][0], landmarks[39][1]) # 左目頭
        right = (landmarks[42][0], landmarks[42][1]) # 右目頭

        face_angle = np.arctan2(right[1] - left[1], right[0] - left[0])
        S = []
        for point in points:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            if -np.pi  <= angle - face_angle <= 0:
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
        print()


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

        
        original_points = get_outline(original_landmarks)

        original_points = np.array(original_points, np.int32)
        original_points = original_points.reshape((-1, 1, 2))
        
        cv2.polylines(image, [original_points], isClosed=True, color=(255, 0, 0), thickness=font_thickness*2)

        # 円を描画
        
        circle_contour = [np.array([p[0]+offset_x, p[1]+offset_y]) for p in circle_contour]


        print(f"offset_x: {offset_x}")
        print(f"offset_y: {offset_y}")
       

        circle_contour = convert_world_to_image(r_matrix, t_vector, camera_matrix, z_point, circle_contour)
        
       
        circle_contour = np.array(circle_contour, np.int32)
        circle_point = circle_contour[0]

        circle_contour = circle_contour.reshape((-1, 1, 2))
        cv2.polylines(image, [circle_contour], isClosed=True, color=(0, 0, 255), thickness=font_thickness*2)

        

        # Roundness Ratioを円の上のテキストとして描画
        text = f"Marugao Score: {marugao_ratio*100:.3f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        circle_center = convert_world_to_image(r_matrix, t_vector, camera_matrix, z_point, [(cx+offset_x, cy+offset_y)])
        cx = circle_center[0][0]
        cy = circle_center[0][1]
        
       
        r = np.sqrt((circle_point[0] - cx) ** 2 + (circle_point[1] - cy) ** 2)
        

       

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
