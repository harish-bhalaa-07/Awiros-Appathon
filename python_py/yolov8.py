import cv2
import numpy as np 
from fetch_data import *


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def postprocess_yolo(outputs,ori_image,dwdh,ratio,names,conf_thresh):
    blobs = []
    image = ori_image.copy()
    conf_thresh = 0.0
    boxes = outputs[:,:4, :]
    confidences = outputs[:,4:, :]
    
    detections = post_process_yolov8(boxes, confidences, confidence_threshold=conf_thresh, nms_threshold=0.4)
    print(detections)
    print("detections: ",len(detections))
    
    # for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(zip(bboxes, confs)):
    for i in range(len(detections)):
        print("i: ",i)
        x0, y0, x1, y1, cls_id, score = detections[i]
        blb = blob()
        
        ori_image_h,ori_image_w,_ =ori_image.shape
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        blb.conf = round(float(score),3)
        name = names[cls_id]

        blb.tx = box[0]
        blb.ty = box[1]
        blb.bx = box[2]
        blb.by = box[3]
        if(blb.tx < 0):
            blb.tx = 0
        if(blb.ty < 0):
            blb.ty < 0
        if(blb.bx > ori_image_w):
            blb.bx = ori_image_w
        if(blb.by > ori_image_h):
            blb.by = ori_image_h

        blb.attribs["Attribute1"] = "Person1"
        blb.attribs["Attribute2"] = "Person2"
        
        blb.id = random.randint(0,1000000)
        blb.label = name
        # print("blob tx: ",blb.tx)
        # print("blob ty: ",blb.ty)
        blb.cropped_frame = ori_image[blb.ty:blb.by, blb.tx:blb.bx, :]

        color = (0,255,0)
        name += ' '+str(blb.conf)
        cv2.rectangle(image,box[:2],box[2:],color,3)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
        blobs.append(blb)
    return image, blobs


def post_process_yolov8(input_boxes, confidences, confidence_threshold=0.5, nms_threshold=0.4):
    # Get the number of classes 1 80 8400
    num_classes = confidences.shape[1]

    # Reshape input_boxes to shape (num_boxes, 4)
    input_boxes = input_boxes[0].T

    # Reshape confidences to shape (num_boxes, num_classes)
    confidences = confidences[0].T

    # Get the indices of boxes with confidence scores above the threshold
    box_indices, class_indices = np.where(confidences > confidence_threshold)

    # Initialize empty lists to store the final detections
    detections = []

    for box_idx, class_idx in zip(box_indices, class_indices):
        # Get the confidence score and class probability for this box
        confidence = confidences[box_idx, class_idx]
        class_prob = confidence * confidences[box_idx, class_idx]

        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = input_boxes[box_idx]

        # Append the detection to the list
        detections.append((x1, y1, x2, y2, class_idx, class_prob))

    # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
    print("detections: ",len(detections))
    if len(detections) > 0:
        detections = np.array(detections)
        keep_indices = nms(detections, nms_threshold)
        detections = detections[keep_indices]

    return detections

# Perform Non-Maximum Suppression (NMS)
def nms(detections, threshold):
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[order[1:]]

        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]

    return keep


def preprocess_yolo(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    return im,ratio, dwdh
