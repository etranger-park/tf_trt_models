import os
from collections import namedtuple

# Change Constancs, Model, CHECKPOINT_PREFIX
MODEL = 'faster_rcnn_resnet101'
CHECKPOINT_PREFIX = 'model.ckpt-200000'
SAVED_BUILD_FILE = True
SAVED_RESULT_IMG = True

CONFIG_FILE = 'pipeline.config'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR,'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')

UNOPTIMIZE_MODEL_DIR = os.path.join(DATA_DIR, 'unoptimized_model')
FROZEN_MODEL_FILE = 'frozen_inference_graph_'+MODEL+'.pb'

TRAINDED_DATA_DIR = os.path.join(DATA_DIR, 'trained_data')
DOWNLOAD_DATA_DIR = os.path.join(DATA_DIR, 'download_data')

CONFIG_PATH = os.path.join(TRAINDED_DATA_DIR, MODEL, CONFIG_FILE)
CHECKPOINT_PATH = os.path.join(TRAINDED_DATA_DIR, MODEL, CHECKPOINT_PREFIX)

DEST_DIR = os.path.join(ROOT_DIR, 'dest')
OPTIMZED_MODEL_DIR = os.path.join(DEST_DIR, 'frozen')
OPTIMZED_MODEL_FILE = MODEL + '_trt.pb'
DEST_IMAGE_DIR = os.path.join(DEST_DIR, 'test_result', MODEL)

DetectionModel = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

INPUT_NAME='image_tensor'
FROZEN_GRAPH_NAME='frozen_inference_graph.pb'

MODELS = {
    'ssd_mobilenet_v1_coco': DetectionModel(
        'ssd_mobilenet_v1_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
        'ssd_mobilenet_v1_coco_2018_01_28',
    ),
    'ssd_mobilenet_v2_coco': DetectionModel(
        'ssd_mobilenet_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        'ssd_mobilenet_v2_coco_2018_03_29',
    ),
    'ssd_inception_v2_coco': DetectionModel(
        'ssd_inception_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
        'ssd_inception_v2_coco_2018_01_28',
    ),
    'ssd_resnet_50_fpn_coco': DetectionModel(
        'ssd_resnet_50_fpn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    ),
    'faster_rcnn_resnet50_coco': DetectionModel(
        'faster_rcnn_resnet50_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        'faster_rcnn_resnet50_coco_2018_01_28',
    ),
    'faster_rcnn_nas': DetectionModel(
        'faster_rcnn_nas',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
        'faster_rcnn_nas_coco_2018_01_28',
    ),
    'mask_rcnn_resnet50_atrous_coco': DetectionModel(
        'mask_rcnn_resnet50_atrous_coco',
        'http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz',
        'mask_rcnn_resnet50_atrous_coco_2018_01_28',
    ),
    'faster_rcnn_inception_v2_coco': DetectionModel(
        'faster_rcnn_inception_v2_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        'faster_rcnn_inception_v2_coco_2018_01_28',
    ),
    'faster_rcnn_resnet101_coco': DetectionModel(
        'faster_rcnn_resnet101_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz',
        'faster_rcnn_resnet101_coco_2018_01_28',
    ),
    'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco': DetectionModel(
        'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz',
        'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28',
    )
}
