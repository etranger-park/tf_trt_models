from PIL import Image
import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from object_detection.utils import visualization_utils as vis_util

from detection import read_optimized_model
from optimize_graph import optimize
from constants import OPTIMZED_MODEL_DIR, \
    OPTIMZED_MODEL_FILE, \
    IMAGE_DIR, \
    SAVED_RESULT_IMG,\
    DEST_IMAGE_DIR

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    # print(image.size)
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def main():
    if not os.path.exists(os.path.join(OPTIMZED_MODEL_DIR, OPTIMZED_MODEL_FILE)):
        optimize()
    trt_graph = read_optimized_model(OPTIMZED_MODEL_FILE, OPTIMZED_MODEL_DIR)

    print("Get the trt graph done")

    # Get the Image path
    TEST_IMAGE_PATHS = sorted(os.listdir(IMAGE_DIR))
    TEST_IMAGE_PATHS = [os.path.join(IMAGE_DIR, filename) for filename in TEST_IMAGE_PATHS]
    category_index = {0:{'name':'background'}, 1:{'name':'blade'}}
    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as tf_sess:
        tf.import_graph_def(trt_graph, name='')
        tf_input = tf_sess.graph.get_tensor_by_name( 'image_tensor:0')
        tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
        tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
        tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
        elapsed_times = []
        for image_path in TEST_IMAGE_PATHS[:100]:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            t0 = time.time()
            (boxes, scores, classes, num) = tf_sess.run(
                [tf_boxes, tf_scores, tf_classes, tf_num_detections],
                feed_dict={tf_input: image_np_expanded})
            t1 = time.time()
            print("elapsed time (ms) : ", float(t1-t0)*1000)
            elapsed_times.append(float(t1-t0)*1000)
            if SAVED_RESULT_IMG:
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                plt.figure()
                plt.imshow(image_np)
                plt.savefig(os.path.join(DEST_IMAGE_DIR, os.path.basename(image_path)))
                plt.close()
            del image
        TRASH = max(elapsed_times) # The First data makes dirty
        print("avg. elapsed time: ", (sum(elapsed_times) - TRASH) / (len(elapsed_times) - 1))

        tf_sess.close()

if __name__ == '__main__':
    main()
