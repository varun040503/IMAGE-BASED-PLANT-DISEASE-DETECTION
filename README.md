# IMAGE-BASED-PLANT-DISEASE-DETECTION
Abstract:
Plant diseases can cause significant damage to crop yields and quality, leading to economic
losses for farmers. Worldwide crop loss is estimated annually to be $220 billion USD or 14% of
crop loss due to plant disease. Crop loss can be caused by biotic organisms which include
oomycetes, fungi, viruses, bacteria, nematodes, as well as abiotic factors like the environment.
Accurate and timely detection of plant diseases is crucial for effective disease management.
Imagebased leaf disease detection has emerged as a promising approach for early and
noninvasive diagnosis of plant diseases. This paper provides a detailed review for imagebased
leaf disease detection using CNN algorithm, including the image acquisition and preprocessing, feature extraction, and classification stages of the detection. The paper also
highlights the challenges and opportunities in this field and identifies areas for future research.
The results of recent studies demonstrate the potential of image-based leaf disease detection for
accurate and automated diagnosis of plant diseases, which could have significant implications
for crop management and food security.
Introduction:
Agriculture is the backbone of our country. It’s a natural and essential science of cultivating the
soil for growing and harvesting crops. Majority of the people, nearly 70%, in our country
depend on the agricultural sector. It’s hard to imagine human life without agriculture. There are
many problems that the farmers face during this process. One of those problems is plant
diseases. Although treating these plants, after detecting the problem, has become far less tedious
now, detecting and recognizing them has become a taxing job. In many rural areas, visual
observation is still the primary approach of disease identification. As farmers and agriculturists
often produce crops on a very large scale, they might fail to identify plant diseases. The Food
and Agriculture Organisation of the United States has estimated that the losses due to diseases
caused by pests alone are between 20% to 40% annually. Technically, the heart of this model is
a machine learning classifier (VGG16) that classifies the obtained images (input) into the
respective plant disease. Many models such as K-Nearest Neighbours (KNN), Support Vector
Machine (SVM), Decision Tree, Logistic Regression (LR), Artificial Neural Networks (ANN),
DeepConvolutional Neural Network (CNN) are available. We used CNN algorithm for disease
detection. Convolutional Neural Networks (CNN) are a type of neural network that are
commonly used for image and video recognition tasks in computer vision. The key feature of a
CNN is its ability to automatically learn spatial hierarchies of features from the input images
through a process called convolution. This means that the network can learn low-level features
like edges, curves, and corners, and then gradually build up to more complex features like
shapes, objects, and scenes. A typical CNN architecture consists of multiple layers, including
convolutional layers, pooling layers, and fully connected layers. The convolutional layers are
responsible for extracting features from the input images, while the pooling layers down sample
the feature maps to reduce their dimensionality. The fully connected layers are used for
classification or regression. In this paper, we used the VGG16 Convolutional Neural Network
model. The VGG16 model has 16 layers, including 13 convolutional layers and 3 fully
connected layers.
LITERATURE SURVEY ON VARIOUS RESEARCH PAPERS:
[1] Identification of plan diseases using CNN and Transfer-Learning approach Hassan, S.M.;
Maji, A.K.; Jasi ´nski, M.; Leonowicz, Z.; Jasi ´nska, E. 2021 Used a CNN model and used
transfer-learning approach
[2] Plant Disease Recognition Based on Image Processing Technology Guiling
Sun,Xinglong,Jia, and Tianyu Geng 2018 Image recognition system based on Multi Linear
Regression is used in this paper.
[3] Image Processing System for Plant Disease Identification by Using FCMClstering
Technique S Megha, R. Niveditha, N Sowmyashree, K.Vidya 2017 an attempt to present
Content Based Image Retrieval (CBIR) system
[4] Analysis of content-based retrieval for plant leaf disease Jayamala K. Patil, Raj Kumar
2017 K-means clustering and SVM are used in this System
[5] Image-Based Detection of Plant Diseases: From Classical Machine Learning to Deep
Learning Journey Rehan Ullah Khan , 1 Khalil Khan , 2 Waleed Albattah , 1 and Ali Mustafa
Qamar 3 2021 Compares how the approach to the same problem has evolved from using the
conventional Machine Learning methods to using Deep learning in the last 5 years.
[6] Using Deeplearning for Image-based plant disease detection Mohanty SP, Hughes DP and
Salathé M 2016 Used AlexNet and GoogLeNet Architecture that are designed in the context of
“Large Scale Visual Recognition Challenge”
[7] Detection of plant leaf disease using Image Segmentation and Soft computing techniques
Vijai Singha , A.K Mishrab 2016 Uses Image Segmentation and several Soft computing
techniques
[8] Transfer learning using VGG-16 model with Deep convolutional neural network for
classifying images Srikanth Tammina, International Journal of Scientific and Research
Publications, Volume 9, Issue 10, October 2019 143 ISSN 2250-3153 2019 Image classification
using VGG16 model.
[9] An Overview of the Research on Plant Leaves Disease detection using Image processing
techniques. Ms. Kiran, R. Gavhale, Prof, Ujwalla Gawande, IOSR Journal of Computer
Engineering 2014 Overview of diseases in plants, main causes and prevention
[10] The Dark Side of Fungi: how they cause diseases in plants Demetrio Marciano, Chiara
Mizzotti, Giuliana Maddalena, Silvia Laura Toffolatti 2021 Provides Very rich knowledge
about the diseases caused by fungi
MODULAR DESIGN AND DESCRIPTION:
The main objective of our research is plant leaf disease detection. We are planning to show the
type of diseases and the affected part of the leaf using image processing as the existing systems
only returns the type of disease that has affected the leaf. We split the dataset into training and
testing datasets in the ratio 80%-training 20%-testing. The input image is processed using
digital image processing techniques like image pre-processing, segmentation. The image is
converted into an array and the segmented database is created. Now the image is classified
using the CNN Classification (using the VGG16 model). If any fault is found, it is displayed
and then the remedies of the disease is displayed on the screen. PROPOSED PROCESS Image
acquisition: Image acquisition involves capturing images of plants using cameras or other
sensors. The quality and resolution of images affect the accuracy of disease detection. We used
the NewPlantDisease Dataset that we obtained from Kaggle which contains about 87k RGB
images of healthy and unhealthy crop leaves which are categorized into 39 different categories.
Pre-processing: It involves filtering and enhancing the images to remove noise and improve
their quality. Various image processing techniques have been applied, including image
segmentation, normalization, and denoising.
PROPOSED PROCESS Image acquisition:
Image acquisition involves capturing images of plants using cameras or other sensors. The
quality and resolution of images affect the accuracy of disease detection. We used the
NewPlantDisease Dataset that we obtained from Kaggle which contains about 87k RGB images
of healthy and unhealthy crop leaves which are categorized into 39 different categories. Preprocessing: It involves filtering and enhancing the images to remove noise and improve their
quality. Various image processing techniques have been applied, including image segmentation,
normalization, and denoising. Feature extraction: Feature extraction involves extracting relevant
features from the images to represent the disease symptoms. Different types of features have
been used, including colour, texture, shape, and spatial features. Classification: Classification
involves training a machine learning model to classify the images into healthy or diseased
plants. Various machine learning techniques have been used, including decision trees, support
vector machines, random forests, deep learning, and convolutional neural networks. Decisionmaking: Decision-making involves using the classification results to make decisions, such as
whether to apply pesticides or other treatments. Decisionmaking can be done manually or
automatically using a feedback loop. DATASET We used the NewPlantDisease dataset that
contains 87k images that are categorised into 39 classes. We included Fungal Diseases and
Bacterial Diseases in plants, as they are the most common type of diseases that occur in plants.
Some of the most common diseases that occur are Fire blight, Leaf spot, Leaf rust, Leaf blight,
Early blight, Late blight, Bacterial Spot, Powdey Mildew. The 39 categories are Apple_scab,
Apple_black_rot, Apple_cedar_apple_rust,Apple_healthy,Ba ckground_without_leaves,
Blueberry_healthy, Cherry_powdey_mildew, Cherry_healthy, Corn_gray_leaf_spot,
Corn_northern_leaf_blight, Corn_healthy, Grape_black_rot, Grape_black_measles,
Grape_leaf_blight, Grape_healthy, Orange_haunglongbing, Peach_bacterial_spot,
Peach_healthy, Pepper_bacterial_spot, Pepper_healthy, Potato_early_blight, Potato_late_blight,
Potato_healthy, Raspberry_healthy, Soybean_healthy, Squash_powdery_mildew,
Strawberry_healthy, Strawberry_leaf_scroch, Tomato_baterial_spot, Tomato_early_blight,
Tomato_healthy, Tomato_late_blight, Tomato_leaf_mold, Tomato_septoria_leaf_spot,
Tomato_spider_mites_twospotted_spider_mite, Tomato_target_spot,Tomato_mosaic_virus ,
Tomato_yellow-leaf_curl_virus.
According to research conducted by Encyclopedia Britannica almost 75% of all the crop
diseases are fungal diseases. This shows us that most of plant diseases are fungal diseases.
Although fungal diseases show symptoms on leaves, Root rot is the most common fungal
disease that affects the root of the plant causing them to decay. Fungi can be very harmful to
plants and can cause significant damage to the plants if not managed properly. Now let us see in
brief about the most common diseases that affect the plants.
It is one of the most common fungal plant diseases. It is caused by a variety of fungal species.
These fungi infect the leaves of plants and create circular spots on the follage. These spots can
vary in colour from yellow, black, brown or even purple mainly depending on the species of the
fungus involved Leaf Blight Leaf Blight appears as irregularly shaped lesions on the leaves of
the plant, which can grow larger and coalesce to form large blighted areas. It is caused by
different fungal species including Bipolaris sorokiniana and Helminthosporium spp. The
affected leaves may turn yellow or brown (in severe cases). The plant may become defoliated,
reducing it’s ability to produce energy through photosynthesis. Fire Blight It infects the tree
through wounds or natural openings such as flowers. Fire blight symptoms include witing,
blackening of leaves and other areas such as twigs, branches. The infected plant parts takes on
scorched appearance and hence the name “Fire Blight”. It is caused by the bacterium Erwinia
amylovora. Leaf Rust It is caused by different fungal species including Puccinia tritcina and
Puccinia recondite. Leaf Rust appears as small yellow-orange pustules on the leaves of the
plant, which may eventually turn brownishblack and produce spores. Powdery Mildew It’s one
of the most common fungal diseases that affects plants. It is caused by different fungal species
including Erysiphe cichoracearum and Sphaerotheca pannosa. It appears as a powdery white or
greyish coating on the leaves and other parts of a plants flowers fruits. The fungal growth can
reduce photosynthesis and cause the affected parts to curl, distort and die.
WORKING OF CNN
1. Input layer - Pixel of image fed as input then the image given in input layer where it accepts
the pixel of the image as input in the form of array.
2. Hidden Layer -Hidden layers carry out feature extraction by performing certain calculation
and manipulation.
● Convolution layer- This layer uses a matrix filter and performs convolution operation to
detect patterns in the image.
● ReLU- ReLU activation function is applied to the convolution layer to get a rectified feature
map of image
● Pooling- Pooling layer also uses multiple filters to detect edges, corners, features etc.
5. Output Layer - Finally there is fully connected layer that identifies the object in the image.
The proposed system takes input image for testing after that pre-processing of image are carried
out. Input image are converted to an array for compression of image according to image size
given. On other side segregated database are created, in this also a pre-processing of data is
done. Image from the segregated database are classified using CNN classification. Input image
after converted to an array after compression, CNN based classification are applied to the
trained dataset (segregated database) and testing dataset (input image) and finally defect of the
leaf are display. If defect is found display disease and Remedy otherwise if defect not found
then leaf is Healthy.
VGG-16 MODEL
VGG-16 is a convolutional neural network (CNN) architecture that was developed by the
Visual Geometry Group at the University of Oxford. The VGG-16 model has 16 layers of
learnable parameters, including 13 convolutional layers and 3 fully connected layers. Each
convolutional layer is followed by a rectified linear activation function (ReLU) and a maxpooling layer. The input to the VGG16 model is an RGB image of size 224x224. The first layer
is a convolutional layer with 64 filters, each of size 3x3, followed by a ReLU activation
function and a max-pooling layer of size 2x2. The second and third convolutional layers have
128 and 256 filters respectively, each of size 3x3, followed by ReLU activation functions and
max-pooling layers of size 2x2. The architecture of the VGG-16 model can be divided into two
parts: the feature extraction part and the classification part. The feature extraction part consists
of the first 13 layers, which extract features from the input image. These layers include
convolutional layers, activation layers, and pooling layers.
It has been trained on large datasets such as ImageNet, which contains over 1 million images
with 1000 different object categories. The VGG-16 model achieved a top-5 error rate of 7.3%
on the ImageNet dataset.
