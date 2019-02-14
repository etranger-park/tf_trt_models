from object_detection.protos import pipeline_pb2
from object_detection.protos import image_resizer_pb2
from object_detection import exporter

import os
import subprocess
import google.protobuf.text_format
import tensorflow as tf

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import replace_relu6 as f_replace_relu6
from graph_utils import remove_assert as f_remove_assert

from constants import MODELS, INPUT_NAME, FROZEN_GRAPH_NAME, \
    CONFIG_FILE,\
    CHECKPOINT_PREFIX, \
    FROZEN_MODEL_FILE, \
    SAVED_BUILD_FILE

INPUT_NAME='image_tensor'
FROZEN_GRAPH_NAME='frozen_inference_graph.pb'

def get_input_names(model):
    return [INPUT_NAME]

def download_detection_model(model, output_dir='.'):
    """Downloads a pre-trained object detection model"""
    print("Download_detection_model %s", model)
    global MODELS

    model_name = model
    try:
        model = MODELS[model_name]
    except:
        assert False, "Can't find in Constants.Model Please Check it"

    subprocess.call(['mkdir', '-p', output_dir])
    tar_file = os.path.join(output_dir, os.path.basename(model.url))

    config_path = os.path.join(output_dir, model.extract_dir, CONFIG_FILE)
    checkpoint_path = os.path.join(output_dir, model.extract_dir, CHECKPOINT_PREFIX)

    if not os.path.exists(os.path.join(output_dir, model.extract_dir)):
        subprocess.call(['wget', model.url, '-O', tar_file])
        subprocess.call(['tar', '-xzf', tar_file, '-C', output_dir])

        # hack fix to handle mobilenet_v2 config bug
        subprocess.call(['sed', '-i', '/batch_norm_trainable/d', config_path])

    return config_path, checkpoint_path

def build_detection_graph(config, checkpoint,
        batch_size=1,
        save_build_file=True,
        score_threshold=None,
        force_nms_cpu=True,
        replace_relu6=True,
        remove_assert=True,
        input_shape=None,
        output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""

    config_path = config
    checkpoint_path = checkpoint
    tmp_dir = os.path.join(output_dir,'.generated_model')
    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(config_path, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppresion.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    # Erase tmp_dir if exist
    if os.path.isdir(tmp_dir):
        subprocess.call(['rm', '-rf', tmp_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                INPUT_NAME,
                config,
                checkpoint_path,
                tmp_dir,
                input_shape=[batch_size, None, None, 3]
            )
    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(tmp_dir, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    if SAVED_BUILD_FILE:
        print("Save build file ...")
        with open(os.path.join(output_dir,FROZEN_MODEL_FILE), 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    # Erase tmp_dir
    if os.path.isdir(tmp_dir):
        subprocess.call(['rm', '-rf', tmp_dir])

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    tf_sess.close()
    return frozen_graph

# for unoptimized model
def read_unoptimized_model(MODEL_FILE, UNOPTIMIZE_MODEL_DIR='../data/unoptimized_model/',
    force_nms_cpu=True,
    replace_relu6=True,
    remove_assert=True
):
    print("Read unoptimized %s" % (MODEL_FILE))
    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(UNOPTIMIZE_MODEL_DIR, MODEL_FILE), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    return frozen_graph

def read_optimized_model(frozen_file, frozen_dir='./dest/frozen/'):
    print("Read optimized %s Model" % (frozen_file))

    trt_graph = tf.GraphDef()
    with open(os.path.join(frozen_dir, frozen_file), 'rb') as f:
        trt_graph.ParseFromString(f.read())

    return trt_graph
