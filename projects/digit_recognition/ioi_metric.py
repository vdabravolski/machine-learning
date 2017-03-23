from keras import backend as K


def iou_metric_func(y_true, y_pred, epsilon=1e-5, sequence_length=5):
    """ Inspired by: http://ronny.rest/tutorials/module/localization_001/intersect_of_union/

        Given two arrays `y_true` and `y_pred` where each row contains a bounding
        boxes for sequence of digits. By default, sequence length is 5. Each digit is represented by 4 numbers:
            [y1, x1, y2, x2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each sequence.

    Args:
        y_true:          (numpy array) sequence_length * each row containing [y1, x1, y2, x2] coordinates
        y_pred:          (numpy array) sequence_length * each row containing [y1, x1, y2, x2] coordinates
        epsilon:    (float) Small value to prevent division by zero
        sequence_length: (int) number of digits in the sequence

    Returns:
        (float) Sum of IoU for all digits in sequence
    """

    y_true = K.reshape(y_true, [-1, 5, 4])
    y_pred = K.reshape(y_pred, [-1, 5, 4])

    y1 = K.maximum(y_true[:, :, 0], y_pred[:, :, 0])
    x1 = K.maximum(y_true[:, :, 1], y_pred[:, :, 1])
    y2 = K.minimum(y_true[:, :, 2], y_pred[:, :, 2])
    x2 = K.minimum(y_true[:, :, 3], y_pred[:, :, 3])

    width = (x2 - x1)
    height = (y2 - y1)

    width = K.clip(width, 0, None)
    height = K.clip(height, 0, None)

    area_overlap = K.tf.multiply(width, height)

    area_a = (y_true[:, :, 2] - y_true[:, :, 0]) * (y_true[:, :, 3] - y_true[:, :, 1])
    area_b = (y_pred[:, :, 2] - y_pred[:, :, 0]) * (y_pred[:, :, 3] - y_pred[:, :, 1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    iou = K.mean(iou)

    return iou
