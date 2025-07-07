from os.path import exists

import torch.autograd.profiler as profiler
import torch
import torch.nn as nn

import numpy as np
from skimage.util import img_as_ubyte
from collections import OrderedDict
import os

import cv2
import time


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def erode_dilate(ff, THR, kernel, iterations):
    """
        Main purpose:
        Both algoritm (erosion & dilay)
        works with boundaries of foreground objects
        """

    """
        Tresholding - convert img 
                      to bitmap 
                      with pointed segmentation 
                      depends on threshold
        """
    _, dets = cv2.threshold(img_as_ubyte(ff), THR, 255, cv2.THRESH_BINARY_INV)

    """ 
        Erode - shrink or erode
                the boundaries of foreground objects
        """
    dets = cv2.erode(dets, kernel, iterations=iterations)

    """ 
        Dilay - expand or dilate
                 the boundaries of foreground objects
        """
    dets = cv2.dilate(dets, kernel, iterations=iterations)

    return dets


def optimized_erode_dilate(ff, THR, kernel, iterations):
    """
    Replace cv2.erode cv2.dilate cv2.treshold
    with np.arr manipulations
    """

    " Treshold "
    dets = (img_as_ubyte(ff) > THR).astype(np.uint8) * 255

    " Erode "
    dets = dets = np.where(np.roll(dets, shift=-1, axis=0), 0, dets)

    " Dilate "
    dets = np.where(np.roll(dets, shift=1, axis=0), 1, dets)

    return dets


def plot_contours(c, original_frames, sigma):
    """
    Plot founded contours
    on image from src

    args:
        c - contours after find_contours
        original_frames - original frame where to plot
        sigma - make bbox wider on sigma constant
    """
    # area testing
    area = cv2.contourArea(c)
    # print(f"area: {area}")

    # place the bbox
    x, y, w, h = cv2.boundingRect(c)

    # find centroid
    # if area > 0
    if area > 0 and area < 250:
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # print(f"check wheteher cx cy are diff {cx, cy}")
        return original_frames, (x, y), (x + w, y + h), (0, 128, 0), x - sigma, y - sigma, w + 2 * sigma, h + 2 * sigma

def save_bbox_to_file(input_path, *args):
    coords = np.array(args)
    with open(input_path, 'a') as file:
        np.savetxt(file, coords, delimiter=' ', newline=' ', fmt='%d')
        file.write('\n')
    file.close()


def is_intersecting(box1, box2):
    #print(f" Exist: {box1}\nCurrent: {box2}")

    x1, y1, x1_w, y1_h = box1
    x2, y2, x2_w, y2_h = box2

    return not (x1_w < x2 or x2_w < x1 or y1_h < y2 or y2_h < y1)


def is_contained(box1, box2):
    x1, y1, x1_w, y1_h = box1
    x2, y2, x2_w, y2_h = box2

    return x2 <= x1 and y2 <= y1 and x2_w >= x1_w and y2_h >= y1_h


def merge_two_boxes(box1, box2):
    x1, y1, x1_w, y1_h = box1
    x2, y2, x2_w, y2_h = box2

    x_merged = min(x1, x2)
    y_merged = min(y1, y2)
    x_merged_w = max(x1_w, x2_w)
    y_merged_h = max(y1_h, y2_h)

    return x_merged, y_merged, x_merged_w, y_merged_h


def join_intersecting_near_boxes(rectangles, threshold=0.2, distance_threshold=5):
    rectangles = np.array(rectangles, dtype=np.int_)
    #print(f"Rect: {rectangles}")

    grouped_rectangles, weights = cv2.groupRectangles(rectangles.tolist(), groupThreshold=1, eps=threshold)

    #print(f"Grouped bboxes: {grouped_rectangles}")

    joined_boxes = []

    if len(grouped_rectangles) > 0:
        boxes = [*grouped_rectangles.tolist()]
    else:
        boxes = rectangles.tolist()

    for i in range(len(boxes)):
        try:
            for j in range(i + 1, len(boxes)):
                if are_boxes_near(boxes[i], boxes[j], distance_threshold):
                    joined_box = merge_boxes(boxes[i], boxes[j])
                    joined_boxes.append(joined_box)
                else:
                    joined_boxes.append(boxes[j])
            # Add the current box if no near boxes were found
            joined_boxes.append(boxes[i])
        except:
            joined_boxes.append(boxes[i])

    #print(f"Joined bboxes: {joined_boxes}")
    return joined_boxes


def extract_xy_wh_coord(x, y, w, h):
    return np.array([[x, y], [x + w, y + h]])


def are_boxes_near(box1, box2, distance_threshold):
    center1 = (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
    center2 = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)

    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return distance <= distance_threshold


def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_merged = min(x1, x2)
    y_merged = min(y1, y2)
    w_merged = max(x1 + w1, x2 + w2) - x_merged
    h_merged = max(y1 + h1, y2 + h2) - y_merged

    return x_merged, y_merged, w_merged, h_merged


def join_boxes_if_near(boxes, distance_threshold):
    joined_boxes = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if are_boxes_near(boxes[i], boxes[j], distance_threshold):
                joined_box = merge_boxes(boxes[i], boxes[j])
                joined_box = cv2.groupRectangles([boxes[i], boxes[j]], groupThreshold=1, eps=distance_threshold)

                joined_boxes.append(joined_box)

    return joined_boxes


def save_bbox(bbox, img_shape, score, cls, save_path, bbox_name):
    """
    Saves a bounding box in YOLO format: score class x_center y_center width height (normalized).

    :param bbox: tuple (x1, y1, x2, y2) — bounding box coordinates
    :param img_shape: tuple (height, width) — image size
    :param score: float — confidence score (between 0.0 and 1.0)
    :param cls: int — class ID
    :param save_path: str — directory where the file will be saved
    :param bbox_name: str — filename (without extension)
    """
    os.makedirs(save_path, exist_ok=True)

    x1, y1, x2, y2 = bbox
    # print(img_shape)
    img_h, img_w, _ = img_shape

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    line = f"{score:.6f} {cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

    # Save to file
    save_file = os.path.join(save_path, f"{bbox_name}.txt")
    with open(save_file, "a") as f:
        f.write(line)


def save_vid(output_path, save_vid_path, fps=30):
    """
    Creates a video from images in the given folder.

    :param output_path: Path to the folder with images.
    :param save_vid_path: Path where the video will be saved.
    :param fps: Frames per second for the output video.
    """
    images = sorted([img for img in os.listdir(output_path) if img.endswith((".png", ".jpg", ".jpeg"))])

    if not images:
        raise ValueError(f"No images found in {output_path}")

    first_image_path = os.path.join(output_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 output
    video = cv2.VideoWriter(save_vid_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(output_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            # print(f"Warning: Unable to read {image_path}")
            continue
        video.write(frame)

    video.release()
    # print(f"Video saved at {save_vid_path}")
