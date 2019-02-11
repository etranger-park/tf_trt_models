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
from tf_trt_models.detection import download_detection_model, build_detection_graph, read_detection_model

print("**TEST Start**")

# from tf_trt_models.detection import download_detection_model,build_detection_graph



MODEL = 'faster_rcnn_inception_v2_coco'
DATA_DIR = './data/'
DEST_DIR = './dest/'
CONFIG_FILE = MODEL + '.config'   # ./data/ssd_inception_v2_coco.config 
CHECKPOINT_FILE = 'model.ckpt'    # ./data/ssd_inception_v2_coco/model.ckpt
IMAGE_PATH = './data/huskies.jpg'
FROZEN_PATH = './dest/frozen/'
# Download the pretrained model
# Download the model configuration file and checkpoint containing pretrained weights by using the following command.
# For improved performance, increase the non-max suppression score threshold in the downloaded config file from 1e-8 to something greater, like 0.1.
'''
config_path, checkpoint_path = download_detection_model(MODEL, 'data')

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    batch_size=1
)
'''
if os.path.exists(os.path.join(FROZEN_PATH, MODEL+'_trt.pb')):
    trt_graph = read_detection_model(MODEL,FROZEN_PATH)
else:
    config_path, checkpoint_path = download_detection_model(MODEL, 'data')

    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        batch_size=1
    )
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with open(os.path.join(DEST_DIR,'frozen',MODEL+'_trt.pb'), 'wb') as f:
        f.write(trt_graph.SerializeToString())

    
print("Get the trt graph done")

# Create session and load graph

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name( 'image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

# Load and Preprocess Image

image = Image.open(IMAGE_PATH)

plt.imshow(image)
plt.savefig(os.path.join(DEST_DIR, 'first.jpg'))
image_resized = np.array(image.resize((300, 300)))
image = np.array(image)
# Run network on Image
print("Try to inference")
scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image_resized[None, ...]
})
print("Inference Done")

boxes = boxes[0] # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = num_detections.astype(int)[0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.imshow(image)

# plot boxes exceeding score threshold
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

    # display rectangle
    patch = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], color='g', alpha=0.3)
    ax.add_patch(patch)

    # display class index and score
    plt.text(x=box[1] + 10, y=box[2] - 10, s='%d (%0.2f) ' % (classes[i], scores[i]), color='w')

# plt.show()
plt.savefig(os.path.join(DEST_DIR, MODEL+'_result.jpg'))

num_samples = 50
print("Checking time .....")
t0 = time.time()
for i in range(num_samples):
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image_resized[None, ...]
    })
t1 = time.time()
print('Average runtime: %f ms' % (float(t1 - t0) * 1000 / num_samples))

tf_sess.close()
