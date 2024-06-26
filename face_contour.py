from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def get_head_contour_points(image, cp='79999_iter.pth'):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    save_pth = cp
    net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.fromarray(image)
    image = img.resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    face_labels = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'hair']

    face_label_nums = [atts.index(label)+1 for label in face_labels]

    # face_labelsの部分のマスクを作成
    face_mask = np.isin(parsing, face_label_nums).astype(np.uint8)

    # 輪郭を検出
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 最大の輪郭を取得
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        return contour.reshape(-1, 2).tolist()
    else:
        return []

def get_face_contour_points(image, cp='79999_iter.pth'):

    
    print("in get_face_contour_points")
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    save_pth = cp
    net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.fromarray(image)
    image = img.resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)

    print("before with torch.no_grad()")
    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    print("after torch.no_grad()")
    
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    face_labels = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'nose', 'mouth', 'u_lip', 'l_lip']

    face_label_nums = [atts.index(label)+1 for label in face_labels]

    print(face_label_nums)
    # face_labelsの部分のマスクを作成
    face_mask = np.isin(parsing, face_label_nums).astype(np.uint8)


    print(face_mask)
    print("findin contours")
    # 輪郭を検出
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    
   
    if contours:
        
        # all_points = np.vstack(contours)

        # # 凸包を計算
        # contour = cv2.convexHull(all_points)
        # 最大の輪郭を取得
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        
        
        return contour.reshape(-1, 2).tolist()
    else:
        return []