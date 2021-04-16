import numpy as np
from scipy.special import softmax

# def np_softmax(x, axis=0):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x, axis=axis))
#     return e_x / e_x.sum(axis=axis)  # only difference


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        self.use_fast_nms = False
        self.max_output_size = 300

    def __call__(self, net_outs, trad_nms=True):
        """
        Args:
             pred_offset: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            pred_cls: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            pred_mask_coef: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            priors: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_out: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.
            Note that the outputs are sorted only if cross_class_nms is False
        """

        box_p = net_outs["pred_offset"].numpy()  # [1, 27429, 4]
        class_p = net_outs["pred_cls"].numpy()  # [1, 27429, 2]
        coef_p = net_outs["pred_mask_coef"].numpy()  # [1, 27429, 32]
        anchors = net_outs["priors"].numpy()  # [27429, 4]
        proto_p = net_outs["proto_out"].numpy()  # [1, 90, 302, 32]

        proto_h = np.shape(proto_p)[1]
        proto_w = np.shape(proto_p)[2]

        box_decode = self._decode(box_p, anchors)  # [1, 27429, 4]

        num_class = np.shape(class_p)[2] - 1

        # Apply softmax to the prediction class
        class_p = softmax(class_p, axis=-1)
        # exclude the background class
        class_p = class_p[:, :, 1:]
        # get the max score class of 27429 predicted boxes
        class_p_max = np.max(class_p, axis=-1)  # [1, 27429]
        batch_size = np.shape(class_p_max)[0]

        # Not using python list here, as tf Autograph has some issues with it
        # https://github.com/tensorflow/tensorflow/issues/37512#issuecomment-600776581
        # detection_boxes = tf.TensorArray(np.float32, size=0, dynamic_size=True)
        # detection_classes = tf.TensorArray(np.float32, size=0, dynamic_size=True)
        # detection_scores = tf.TensorArray(np.float32, size=0, dynamic_size=True)
        # detection_masks = tf.TensorArray(np.float32, size=0, dynamic_size=True)
        # num_detections = tf.TensorArray(np.int32, size=0, dynamic_size=True)

        detection_boxes = []
        detection_classes = []
        detection_scores = []
        detection_masks = []
        num_detections = []

        for b in range(batch_size):
            # filter predicted boxes according the class score
            class_thre = class_p[b][np.where(class_p_max[b] > 0.3)]
            box_thre = box_decode[b][np.where(class_p_max[b] > 0.3)]
            coef_thre = coef_p[b][np.where(class_p_max[b] > 0.3)]

            if len(class_thre) == 0:
                # TODO: Check this
                detection_boxes.append(np.zeros((self.max_output_size, 4)))
                detection_classes.append(np.zeros((self.max_output_size)))
                detection_scores.append(np.zeros((self.max_output_size)))
                detection_masks.append(
                    np.zeros((self.max_output_size, proto_h, proto_w)),
                )
                num_detections.append(0)
            else:
                if not trad_nms:
                    box_thre, coef_thre, class_ids, class_thre = self._traditional_nms(
                        box_thre, coef_thre, class_thre
                    )
                else:
                    box_thre, coef_thre, class_ids, class_thre = self._traditional_nms(
                        box_thre, coef_thre, class_thre
                    )

                # Padding with zeroes to reach max_output_size
                class_ids = np.concatenate(
                    [class_ids, np.zeros(self.max_output_size - np.shape(box_thre)[0])],
                    0,
                )
                class_thre = np.concatenate(
                    [
                        class_thre,
                        np.zeros(self.max_output_size - np.shape(box_thre)[0]),
                    ],
                    0,
                )
                num_detection = np.shape(box_thre)[0]
                pad_num_detection = self.max_output_size - num_detection

                _masks_coef = np.matmul(proto_p[b], np.transpose(coef_thre))
                _masks_coef = np_sigmoid(_masks_coef)  # [138, 138, NUM_BOX]

                boxes, masks = self._sanitize(_masks_coef, box_thre)
                masks = np.transpose(masks, (2, 0, 1))
                paddings = np.array([[0, pad_num_detection], [0, 0], [0, 0]])
                masks = np.pad(masks, paddings, "constant")

                paddings = np.array([[0, pad_num_detection], [0, 0]])
                boxes = np.pad(boxes, paddings, "constant")

                detection_boxes.append(boxes[np.newaxis, ...])
                detection_classes.append(class_ids)
                detection_scores.append(class_thre)
                detection_masks.append(masks[np.newaxis, ...])
                num_detections.append(num_detection)

        detection_boxes = np.concatenate(detection_boxes)
        # detection_classes = np.concatenate(detection_classes)
        # detection_scores = np.concatenate(detection_scores)
        detection_masks = np.concatenate(detection_masks)
        num_detections = np.array(num_detections)
        # num_detections = np.concatenate(num_detections)

        result = {
            "detection_boxes": detection_boxes,
            "detection_classes": detection_classes,
            "detection_scores": detection_scores,
            "detection_masks": detection_masks,
            "num_detections": num_detections,
        }
        return result

    def _decode(self, box_p, priors, include_variances=False):
        # https://github.com/feiyuhuahuo/Yolact_minimal/blob/9299a0cf346e455d672fadd796ac748871ba85e4/utils/box_utils.py#L151
        """
        Decode predicted bbox coordinates using the scheme
        employed at https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
            b_x = prior_w*loc_x + prior_x
            b_y = prior_h*loc_y + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)

        Note that loc is inputed as [c_x, x_y, w, h]
        while priors are inputed as [c_x, c_y, w, h] where each coordinate
        is relative to size of the image.

        Also note that prior_x and prior_y are center coordinates.
        """
        variances = [0.1, 0.2]
        # box_p = tf.cast(box_p, tf.float32)
        # priors = tf.cast(priors, tf.float32)
        if include_variances:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * priors[:, 2:] * variances[0]
            b_w_h = priors[:, 2:] * np.exp(box_p[:, :, 2:] * variances[1])
        else:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * priors[:, 2:]
            b_w_h = priors[:, 2:] * np.exp(box_p[:, :, 2:])

        boxes = np.concatenate([b_x_y, b_w_h], axis=-1)

        # [x_min, y_min, x_max, y_max]
        boxes = np.concatenate(
            [
                boxes[:, :, :2] - boxes[:, :, 2:] / 2,
                boxes[:, :, 2:] / 2 + boxes[:, :, :2],
            ],
            axis=-1,
        )

        # [y_min, x_min, y_max, x_max]
        return np.stack(
            [boxes[:, :, 1], boxes[:, :, 0], boxes[:, :, 3], boxes[:, :, 2]], axis=-1
        )

    def _sanitize_coordinates(self, _x1, _x2, padding: int = 0):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.
        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        x1 = np.minimum(_x1, _x2)
        x2 = np.maximum(_x1, _x2)
        x1 = np.clip(x1 - padding, 0, 1)
        x2 = np.clip(x2 + padding, 0, 1)

        # Normalize the coordinates
        return x1, x2

    def _sanitize(self, masks, boxes, padding: int = 0, crop_size=(30, 30)):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        # h, w, n = masks.shape

        x1, x2 = self._sanitize_coordinates(boxes[:, 1], boxes[:, 3], padding)
        y1, y2 = self._sanitize_coordinates(boxes[:, 0], boxes[:, 2], padding)

        # Making adjustments for tf.image.crop_and_resize
        boxes = np.stack((y1, x1, y2, x2), axis=1)

        # box_indices = tf.zeros(tf.shape(boxes)[0], dtype=tf.int32) # All the boxes belong to a single batch
        # masks = tf.expand_dims(tf.transpose(masks, (2,0,1)), axis=-1)
        # masks = tf.image.crop_and_resize(masks, boxes, box_indices, crop_size)

        return boxes, masks

    def _traditional_nms(
        self,
        boxes,
        masks,
        scores,
        iou_threshold=0.5,
        score_threshold=0.3,
        max_class_output_size=100,
        max_output_size=300,
        soft_nms_sigma=0.5,
    ):
        num_classes = np.shape(scores)[1]
        # List won't work as for now
        # https://github.com/tensorflow/tensorflow/issues/37512#issuecomment-600776581
        # box_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # mask_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # cls_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # scr_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        box_lst_arr = []
        mask_lst_arr = []
        cls_lst_arr = []
        scr_lst_arr = []

        # for _cls in range(num_classes):
        #     cls_scores = scores[:, _cls]
        #     (
        #         selected_indices,
        #         selected_scores,
        #     ) = nms(
        #         boxes,
        #         cls_scores,
        #         iou_threshold
        #     )

        #     box_lst_arr.append(boxes[selected_indices])
        #     mask_lst_arr.append(masks[selected_indices])
        #     cls_lst_arr.append(cls_scores[selected_indices] * 0.0 + _cls + 1.0)  # class ID starting from 1
        #     scr_lst_arr.append(cls_scores[selected_indices])

        # print(boxes)
        best_bboxes, selected_indices = nms(
            np.concatenate(
                [
                    boxes,
                    np.max(scores, axis=-1)[..., np.newaxis],
                    np.argmax(scores, axis=-1)[..., np.newaxis] + 1,
                ],
                axis=-1,
            ),
            iou_threshold=0.5,
        )
        # print(best_bboxes, selected_indices)
        # boxes = np.concatenate(box_lst_arr)
        # masks = np.concatenate(mask_lst_arr)
        # classes = np.concatenate(cls_lst_arr)
        # scores = np.concatenate(scr_lst_arr)
        boxes = boxes[selected_indices][:max_output_size]
        masks = masks[selected_indices][:max_output_size]
        classes = np.argmax(scores, axis=-1)[selected_indices][:max_output_size] + 1
        scores = np.max(scores, axis=-1)[selected_indices][:max_output_size]

        # _ids = np.argsort(-1 * scores)
        # scores = scores[_ids][:max_output_size]
        # boxes = boxes[_ids][:max_output_size]
        # masks = masks[_ids][:max_output_size]
        # classes = classes[_ids][:max_output_size]

        return boxes, masks, classes, scores


def _nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_indices = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_indices.append(index)
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_indices, picked_score


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method="nms"):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    best_indices = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]
        _indices = np.where(cls_mask)[0]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            best_indices.append(_indices[max_ind])
            # _indices = np.delete(_indices, max_ind, 0)
            # cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ["nms", "soft-nms"]

            if method == "nms":
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
                # print(weight)

            if method == "soft-nms":
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]
            _indices = _indices[score_mask]

    return best_bboxes, best_indices
