import sys
import os
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from detection import download_detection_model, build_detection_graph, read_unoptimized_model, read_optimized_model
from constants import UNOPTIMIZE_MODEL_DIR, MODEL, CONFIG_PATH, \
    CHECKPOINT_PATH, \
    DOWNLOAD_DATA_DIR, \
    OPTIMZED_MODEL_DIR, \
    OPTIMZED_MODEL_FILE, \
    FROZEN_MODEL_FILE, \
    PRECISION_MODE

BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'

def optimize():
    print("\nOptimize %s Start\n" % MODEL)
    if os.path.exists(os.path.join(UNOPTIMIZE_MODEL_DIR, FROZEN_MODEL_FILE)):
        # Exist unoptimized Build Model
        frozen_graph = read_unoptimized_model(FROZEN_MODEL_FILE, UNOPTIMIZE_MODEL_DIR=UNOPTIMIZE_MODEL_DIR)
    else:
        if os.path.exists(CONFIG_PATH):
            # Using Own Trained Data
            config_path = CONFIG_PATH
            checkpoint_path = CHECKPOINT_PATH
        else:
            # Download from tensorflow coco site
            config_path, checkpoint_path = download_detection_model(MODEL, DOWNLOAD_DATA_DIR)
        print("Build Model ...")
        frozen_graph = build_detection_graph(
                config=config_path,
                checkpoint=checkpoint_path,
                batch_size=1,
                output_dir=UNOPTIMIZE_MODEL_DIR,
                force_nms_cpu=True
            )
        print("Build Done")

    print("Optimizing....")
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=output_names,
                max_batch_size=1,
                maximum_cached_engines=5,
                max_workspace_size_bytes=1 << 30,
                precision_mode=PRECISION_MODE,
                minimum_segment_size=150
            )
        with open(os.path.join(OPTIMZED_MODEL_DIR, OPTIMZED_MODEL_FILE), 'wb') as f:
            f.write(trt_graph.SerializeToString())

    print("\n\nOptimize done, trt.pb file is in dest/frozen\n\n")
    tf_sess.close()


if __name__ == '__main__':
    optimize()
