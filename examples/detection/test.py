from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph, read_trained_model, read_optimized_model
from object_detection.utils import visualization_utils as vis_util

print("**TEST Start**")

# from tf_trt_models.detection import download_detection_model,build_detection_graph


BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'

USED_TRAINED_DATA = True
SAVED_RESULT = False
MODEL = 'faster_rcnn_resnet101'
DATA_DIR = './data/'
DEST_DIR = './dest/'
CONFIG_FILE = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt-200000'
IMAGE_DIR_PATH = os.path.join(DATA_DIR,'images')
TRAINDED_DIR_PATH = os.path.join(DATA_DIR, 'trained_model')
OPTIMIZED_PATH = os.path.join(DEST_DIR, 'frozen')

category_index = {0:{'name':'background'}, 1:{'name':'blade'}}
# Download the pretrained model
# Download the model configuration file and checkpoint containing pretrained weights by using the following command.
# For improved performance, increase the non-max suppression score threshold in the downloaded config file from 1e-8 to something greater, like 0.1.
if __name__ == '__main__':
    if os.path.exists(os.path.join(OPTIMIZED_PATH, MODEL+'_trt.pb')):
        # already optimzed the model
        trt_graph = read_optimized_model(MODEL,OPTIMIZED_PATH)
    else:
        if os.path.exists(os.path.join(TRAINDED_DIR_PATH, 'frozen_inference_graph_'+MODEL+'.pb')):
            # trained Model not optimized
            TRAINED_MODEL = 'frozen_inference_graph_'+MODEL+'.pb'
            frozen_graph = read_trained_model(TRAINED_MODEL, TRAINDED_DIR_PATH)
        else:
            if USED_TRAINED_DATA:
                print("Using trained Data")
                config_path = os.path.join(DATA_DIR,'trained_data',MODEL,CONFIG_FILE)
                checkpoint_path = os.path.join(DATA_DIR,'trained_data',MODEL,CHECKPOINT_PREFIX)
            else :
                config_path, checkpoint_path = download_detection_model(MODEL, 'data')
            frozen_graph = build_detection_graph(
                config=config_path,
                checkpoint=checkpoint_path,
                batch_size=1,
                force_nms_cpu=True
            )
        output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            maximum_cached_engines=3,
            max_workspace_size_bytes=1 << 20,
            precision_mode='FP32',
            minimum_segment_size=20
        )
        with open(os.path.join(DEST_DIR,'frozen',MODEL+'_trt.pb'), 'wb') as f:
            f.write(trt_graph.SerializeToString())

    print("Get the trt graph done")

    # Create session and load graph
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        # print(image.size)
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    TEST_IMAGE_PATHS = sorted(os.listdir(IMAGE_DIR_PATH))
    TEST_IMAGE_PATHS = [os.path.join(IMAGE_DIR_PATH, filename) for filename in TEST_IMAGE_PATHS]
    # Size, in inches, of the output images.


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
        print("Total image number %d" % len(TEST_IMAGE_PATHS))
        for image_path in TEST_IMAGE_PATHS[:100]:
            image = Image.open(image_path)
            # Load and Preprocess Image
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
            # print("classes",np.squeeze(classes).astype(np.int32))
            # print("num detection",num)
            if SAVED_RESULT:
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
                plt.savefig('dest/test_result/%s/%s' % (MODEL,os.path.basename(image_path)))
                print("saved %s" % os.path.basename(image_path))
                plt.close()
            del image
        TRASH = max(elapsed_times) # The First data makes dirty
        print("avg. elapsed time: ", (sum(elapsed_times) - TRASH) / (len(elapsed_times) - 1))

        tf_sess.close()
