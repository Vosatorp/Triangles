#!/usr/bin/env python3
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract
import random
import os
import argparse
from collections import Counter

import warnings

warnings.filterwarnings("ignore")

from shapely.geometry import Polygon


TRANSFORMED_SIZE = 100
PADDING = 0
DEBUG_PREVIEW_SIZE = 500


def show_and_exit(img):
    print(type(img))
    scaled = cv2.resize(img, (500, 500))

    cv2.imshow("Image", scaled)
    key = cv2.waitKey(0)
    if key == ord('q'):
        return True
    return False


def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def score_triangle(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p3 - p2

    angle1 = angle_between_vectors(v1, v2)
    angle2 = angle_between_vectors(-v1, v3)
    angle3 = angle_between_vectors(-v2, -v3)

    if angle1 == np.nan or angle2 == np.nan or angle3 == np.nan:
        return 10 ** 10

    return max(angle1, angle2, angle3)


def triangle_iou(triangle1, triangle2):
    poly1 = Polygon(triangle1)
    poly2 = Polygon(triangle2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union
    return iou


def get_triangle(mask):
    points = mask[0].xy[0]

    p1 = points[0]
    f1 = f2 = f3 = max(points, key=lambda p: np.linalg.norm(p - p1))

    for _ in range(2):
        f1 = max(points, key=lambda p: np.linalg.norm(p - f3) + np.linalg.norm(p - f2))
        f2 = max(points, key=lambda p: np.linalg.norm(p - f1) + np.linalg.norm(p - f3))
        f3 = max(points, key=lambda p: np.linalg.norm(p - f1) + np.linalg.norm(p - f2))

    return f1, f2, f3


def get_affines(p1, p2, p3):
    if np.cross(p2 - p1, p3 - p1) > 0:
        p2, p3 = p3, p2

    dest_points = np.float32([[0, 1], [1, 1], [0.5, 0.2]]) * 100

    yield cv2.getAffineTransform(np.float32([p1, p2, p3]), dest_points)
    yield cv2.getAffineTransform(np.float32([p2, p3, p1]), dest_points)
    yield cv2.getAffineTransform(np.float32([p3, p1, p2]), dest_points)


def filter_components(img_crop, min_area=50, max_area=500):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_crop)
    mask = np.zeros(img_crop.shape, dtype=np.uint8)
    max_area_component = None
    max_area_value = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area_value and min_area < area < max_area:
            max_area_value = area
            max_area_component = i

    if max_area_component is not None:
        component_mask = (labels == max_area_component).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component_mask)

    return mask, max_area_value



def get_OCR(reader, img_new, max_val, K):
    img_new = cv2.threshold(img_new, max_val - K, 255, cv2.THRESH_BINARY)[1]

    img_new, area = filter_components(img_new, 120, 520)

    if area is None:
        return None, None

    kernel = np.ones((3, 3), np.uint8)
    img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, kernel)
    img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, kernel)

    bb = cv2.boundingRect(img_new)
    img_new = 255 - img_new

    img_crop = img_new[
               max(0, bb[1] - 10):min(bb[1] + bb[3] + 10, img_new.shape[0] - 1),
               max(0, bb[0] - 10):min(bb[0] + bb[2] + 10, img_new.shape[1] - 1)
               ]

    text = reader.readtext(img_crop, allowlist="12345678")

    # Pytesseract
    best_num_pt = None

    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=12345678'
    text_pt = pytesseract.image_to_string(img_crop, config=custom_config).strip()
    if text_pt and len(text_pt) == 1:
        best_num_pt = int(text_pt)
        if not (1 <= best_num_pt <= 8):
            best_num_pt = None

    best_num = None
    best_prob = 0
    for box, num, prob in text:
        print(box, num, prob)
        if prob > best_prob and prob > 0.5 and len(num) == 1:
            best_num = int(num)
            best_prob = prob

    print(f"best_num: {best_num}\t best_pt: {best_num_pt}\t K: {K}  AREA: {area}")

    return best_num, best_num_pt


def parse_img(res_img, dir, imshow, ground_truth):
    reader = easyocr.Reader(['en'])

    annotated_img = res_img.orig_img.copy()
    os.makedirs(f"./{dir}", exist_ok=True)
    result = []
    coordinates = []

    for object in res_img:
        mask = object.masks

        img = cv2.cvtColor(res_img.orig_img, cv2.COLOR_BGR2GRAY)

        p1, p2, p3 = get_triangle(mask)
        print("Triangle:", p1, p2, p3)

        print("Score: ", score_triangle(p1, p2, p3))

        crop_mask = cv2.fillPoly(np.zeros(img.shape, dtype=img.dtype), [np.array([p1, p2, p3], np.int32)], (1, 1, 1))

        avg = np.mean(img[crop_mask == 1])
        max_val = np.max(img[crop_mask == 1])
        img = img * crop_mask + np.uint8(avg) * (1 - crop_mask)

        resulting_imgs = []
        num_counter = Counter()

        if score_triangle(p1, p2, p3) < 80:  # Проверка на то что треугольник более-менее равносторонний
            for K in [50, 70, 90, 60, 80, 100]:
                for affine in get_affines(p1, p2, p3):
                    img_new = cv2.warpAffine(img, affine, (100, 100), borderValue=avg)
                    best_num, best_num_pt = get_OCR(reader, img_new, max_val, K)
                    if best_num:
                        num_counter[best_num] += 1
                    if best_num_pt:
                        num_counter[best_num_pt] += 1
                    if best_num == best_num_pt and best_num is not None:
                        num_counter[best_num] += 3
                        num_counter[best_num_pt] += 3
                        break
                if K == 100 and len(num_counter) == 1:
                    break

        print("COUNTER:", num_counter)
        best_num = num_counter.most_common(1)[0][0] if num_counter else None

        cv2.polylines(annotated_img, [np.array([p1, p2, p3], np.int32)], True, (0, 255, 0), 2)
        centroid = np.mean(np.array([p1, p2, p3]), axis=0).astype(int)

        if best_num:
            cv2.putText(annotated_img, str(best_num), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            result.append(int(best_num))

        coordinates.append((tuple(p1), tuple(p2), tuple(p3), int(best_num) if best_num is not None else None))

        for polygon in mask.xy:
            polygon = np.array(polygon, dtype=np.int32)
            cv2.polylines(annotated_img, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        if imshow:
            cv2.imshow("Annotated Image", annotated_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                return

    if imshow:
        cv2.imshow("Annotated Image", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(f"./{dir}/annotated_image.png", annotated_img)

    print("Dice count: ", len(result))
    print("Result:", result)
    for coord in coordinates:
        print(f"Triangle: {coord[:3]}, Number: {coord[3]}")

    with open(f"./{dir}/result.txt", "w") as f:
        f.write(f"Dice count: {len(result)}\n")
        for coord in coordinates:
            f.write(f"Triangle: {coord[:3]}, Number: {coord[3]}\n")

    if ground_truth:
        calc_f1()


def main():
    parser = argparse.ArgumentParser(description="Run YOLO model on a list of photos.")
    parser.add_argument('--checkpoint', type=str, default='best.pt', help='Path to the YOLO checkpoint file')
    parser.add_argument('--photos', type=str, nargs='+', required=True, help='List of photo paths to process')
    parser.add_argument('--test_dir', type=str, nargs='+', required=True)
    parser.add_argument('--imshow', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth file for F1 score calculation')
    args = parser.parse_args()

    model = YOLO(args.checkpoint).to(args.device)
    results = model(args.photos)

    for res_img, folder in zip(results, args.test_dir):
        parse_img(res_img, folder, args.imshow, args.ground_truth)


if __name__ == "__main__":
    main()
