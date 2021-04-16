import base64
import json
import os
import sys
import time

import tensorflow as tf
import numpy as np
import random
import colorsys
import redis
import cv2
from PIL import Image
import io

from detection_numpy import Detect

REDIS_HOST = os.environ.get("REDIS_HOST")
IMAGE_QUEUE = os.environ.get("IMAGE_QUEUE")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))

IMAGE_DTYPE = os.environ.get("IMAGE_DTYPE")
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))
IMAGE_CHANS = int(os.environ.get("IMAGE_CHANS"))

SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP"))

# Connect to Redis server
db = redis.StrictRedis(host=REDIS_HOST)

# Load the pre-trained tf saved model
model = tf.saved_model.load("/app/saved_model_0.7416321_35000")
detect = Detect(num_classes=9, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)


def base64_decode_image(a, dtype, shape):
    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # Convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # Return the decoded image
    return a.astype(np.float32)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def prepare_image(image, target):
    x = cv2.resize(image, target)  # If model was trained with 550x550 size images.
    x = x[np.newaxis, ...]  # Since infering on a single image, bactch size will be 1.
    return x.astype(np.float32)


def predict(img):
    colors_list = random_colors(200)
    pred_cls, pred_offset, pred_mask_coef, protonet_output, seg_out, priors = model(
        prepare_image(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    )
    output = {
        "pred_cls": pred_cls,
        "pred_offset": pred_offset,
        "pred_mask_coef": pred_mask_coef,
        "proto_out": protonet_output,
        "seg": seg_out,
        "priors": priors,
    }
    output.update(detect(output))
    _h = img.shape[0]
    _w = img.shape[1]

    det_num = output["num_detections"][0]
    det_boxes = output["detection_boxes"][0][:det_num]
    if not len(det_boxes) == 0:
        det_boxes = det_boxes * np.array([_h, _w, _h, _w])
    det_masks = output["detection_masks"][0][:det_num]

    det_scores = output["detection_scores"][0][:det_num]
    det_classes = output["detection_classes"][0][:det_num]

    class_names = {5: "rust", 7: "scale", 8: "oil_leaking"}
    for i in range(det_num):
        score = det_scores[i]
        if score > 0.1:
            box = det_boxes[i].astype(int)
            _class = det_classes[i]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), colors_list[i], 2)
            cv2.putText(
                img,
                class_names[_class] + "; " + str(round(score, 2)),
                (box[1], box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors_list[i],
                lineType=cv2.LINE_AA,
            )

            mask = det_masks[i]
            mask = cv2.resize(mask, (_w, _h))
            mask = mask > 0.1
            roi = img[mask]
            blended = roi.astype("uint8")
            img[mask] = blended * colors_list[i]

    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return base64.b64encode(rawBytes.read()).decode("ascii")


def classify_process():
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            pipe.ltrim(IMAGE_QUEUE, BATCH_SIZE, -1)
            queue, _ = pipe.execute()

        imageIDs = []
        batch = None
        for q in queue:
            # Deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(
                q["image"], IMAGE_DTYPE, (1, q["height"], q["width"], 3)
            )

            # Check to see if the batch list is None
            if batch is None:
                batch = image

            # Otherwise, stack the data
            else:
                batch = np.vstack([batch, image])

            # Update the list of image IDs
            imageIDs.append(q["id"])

        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            results = []
            for i, img in enumerate(batch):
                results.append([predict(img), img.shape[0], img.shape[1]])

            # Loop over the image IDs and their corresponding set of results from our model
            for (imageID, resultImage) in zip(imageIDs, results):
                # Initialize the list of output predictions
                output = []

                # Loop over the results and add them to the list of output predictions
                r = {
                    "result": resultImage[0],
                    "height": resultImage[1],
                    "width": resultImage[2],
                }
                # Store the output predictions in the database, using image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(r))

        # Sleep for a small amount
        time.sleep(SERVER_SLEEP)


if __name__ == "__main__":
    classify_process()
