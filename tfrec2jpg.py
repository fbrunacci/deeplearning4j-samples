import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt
from IPython.display import display, Image as Img
from PIL import Image 
import glob
import io

work_dir = '/home/fabien/.deeplearning4j/data/ISIC_2020/Deotte/malignant-v2-512x512'
swd = f'{work_dir}/tfrec2jpeg'

# Create a dictionary describing the features.
image_feature_description = {
    'age_approx': tf.io.FixedLenFeature([], tf.int64),
    'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    'diagnosis': tf.io.FixedLenFeature([], tf.int64),
#    'height': tf.io.FixedLenFeature([], tf.int64),
#    'width': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'patient_id': tf.io.FixedLenFeature([], tf.int64),
    'sex': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature([], tf.int64),
}


tfrecFiles = glob.glob(f'{work_dir}/*.tfrec')
for tfrec in tfrecFiles:
    print(f"{tfrec}")
    raw_dataset = tf.data.TFRecordDataset(tfrec)
    for raw_record in raw_dataset:
        features = tf.io.parse_single_example(raw_record, image_feature_description)
        image_name = features['image_name'].numpy().decode('UTF-8') + ".jpg"
        print(f"{image_name}")
#        width = features['width'].numpy()
#        height = features['height'].numpy()
#        print(f"{image_name} {width}x{height}")
        image_raw = features['image'].numpy()
        image = Image.open(io.BytesIO(image_raw))
        image.save(f"{swd}/{image_name}")
        #display(Img(data=image_raw))

