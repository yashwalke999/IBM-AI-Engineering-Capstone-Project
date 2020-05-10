# IBM-AI-Engineering-Capstone-Project
Keras - Evaluating and Comparing Pretrained Models

Classification models Zoo - Keras (and TensorFlow Keras)
Trained on ImageNet classification models. The library is designed to work both with Keras and TensorFlow Keras. See example below.

Important!
There was a huge library update 05 of August. Now classification-models works with both frameworks: keras and tensorflow.keras. If you have models, trained before that date, to load them, please, use image-classifiers (PyPI package name) of 0.2.2 version. You can roll back using pip install -U image-classifiers==0.2.2.

Architectures:
VGG [16, 19]
ResNet [18, 34, 50, 101, 152]
ResNeXt [50, 101]
SE-ResNet [18, 34, 50, 101, 152]
SE-ResNeXt [50, 101]
SE-Net [154]
DenseNet [121, 169, 201]
Inception ResNet V2
Inception V3
Xception
NASNet [large, mobile]
MobileNet
MobileNet v2
Specification
The top-k accuracy were obtained using center single crop on the 2012 ILSVRC ImageNet validation set and may differ from the original ones. The input size used was 224x224 (min size 256) for all models except:

NASNetLarge 331x331 (352)
InceptionV3 299x299 (324)
InceptionResNetV2 299x299 (324)
Xception 299x299 (324)
The inference *Time was evaluated on 500 batches of size 16. All models have been tested using same hardware and software. Time is listed just for comparison of performance.

Model	Acc@1	Acc@5	Time*	Source
vgg16	70.79	89.74	24.95	keras
vgg19	70.89	89.69	24.95	keras
resnet18	68.24	88.49	16.07	mxnet
resnet34	72.17	90.74	17.37	mxnet
resnet50	74.81	92.38	22.62	mxnet
resnet101	76.58	93.10	33.03	mxnet
resnet152	76.66	93.08	42.37	mxnet
resnet50v2	69.73	89.31	19.56	keras
resnet101v2	71.93	90.41	28.80	keras
resnet152v2	72.29	90.61	41.09	keras
resnext50	77.36	93.48	37.57	keras
resnext101	78.48	94.00	60.07	keras
densenet121	74.67	92.04	27.66	keras
densenet169	75.85	92.93	33.71	keras
densenet201	77.13	93.43	42.40	keras
inceptionv3	77.55	93.48	38.94	keras
xception	78.87	94.20	42.18	keras
inceptionresnetv2	80.03	94.89	54.77	keras
seresnet18	69.41	88.84	20.19	pytorch
seresnet34	72.60	90.91	22.20	pytorch
seresnet50	76.44	93.02	23.64	pytorch
seresnet101	77.92	94.00	32.55	pytorch
seresnet152	78.34	94.08	47.88	pytorch
seresnext50	78.74	94.30	38.29	pytorch
seresnext101	79.88	94.87	62.80	pytorch
senet154	81.06	95.24	137.36	pytorch
nasnetlarge	82.12	95.72	116.53	keras
nasnetmobile	74.04	91.54	27.73	keras
mobilenet	70.36	89.39	15.50	keras
mobilenetv2	71.63	90.35	18.31	keras
Weights
Name	Classes	Models
'imagenet'	1000	all models
'imagenet11k-place365ch'	11586	resnet50
'imagenet11k'	11221	resnet152
Installation
Requirements:

Keras >= 2.2.0 / TensorFlow >= 1.12
keras_applications >= 1.0.7
Note
This library does not have TensorFlow in a requirements for installation. 
Please, choose suitable version (‘cpu’/’gpu’) and install it manually using 
official Guide (https://www.tensorflow.org/install/).
PyPI stable package:

$ pip install image-classifiers==0.2.2
PyPI latest package:

$ pip install image-classifiers==1.0.0b1
Latest version:

$ pip install git+https://github.com/qubvel/classification_models.git
Examples
Loading model with imagenet weights:
# for keras
from classification_models.keras import Classifiers

# for tensorflow.keras
# from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18((224, 224, 3), weights='imagenet')
This way take one additional line of code, however if you would like to train several models you do not need to import them directly, just access everything through Classifiers.

You can get all model names using Classifiers.models_names() method.

Inference example:
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

# read and prepare image
x = imread('./imgs/tests/seagull.jpg')
x = resize(x, (224, 224)) * 255    # cast back to 0-255 range
x = preprocess_input(x)
x = np.expand_dims(x, 0)

# load model
model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)

# processing image
y = model.predict(x)

# result
print(decode_predictions(y))
Model fine-tuning example:
import keras
from classification_models.keras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

# prepare your data
X = ...
y = ...

X = preprocess_input(X)

n_classes = 10

# build model
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y)
