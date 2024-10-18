<h1>🐕 Object Detection with YOLOv5 🐈</h1>

<h2>Overview</h2>
<p>Welcome to the <strong>Object Detection with YOLOv5</strong> project! This project utilizes the powerful YOLOv5 model to detect various objects, including animals, food items, and household objects. The primary goal of this project is to train a YOLOv5 model to identify and classify multiple objects in images. We have prepared a custom dataset consisting of various classes, including:</p>
<ul>
    <li>Dog</li>
    <li>Cat</li>
    <li>TV</li>
    <li>Car</li>
    <li>Meatballs</li>
    <li>Marinara Sauce</li>
    <li>Tomato Soup</li>
    <li>Chicken Noodle Soup</li>
    <li>French Onion Soup</li>
    <li>Chicken Breast</li>
    <li>Ribs</li>
    <li>Pulled Pork</li>
    <li>Hamburger</li>
    <li>Cavity</li>
    <li><strong>😴 Awake</strong></li>
    <li><strong>😌 Drowsy</strong></li>
</ul>
<p>This model can be used for various applications, including surveillance, inventory management, and more! 🚀</p>

<h2>Table of Contents</h2>
<ol>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#dataset-preparation">Dataset Preparation</a></li>
    <li><a href="#image-capture-for-dataset">Image Capture for Dataset</a></li>
    <li><a href="#labeling-with-labelimg">Labeling with LabelImg</a></li>
    <li><a href="#training-the-model">Training the Model</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
</ol>

<h2 id="requirements">Requirements</h2>
<p>Before you begin, ensure you have the following libraries installed:</p>
<ul>
    <li><strong>Python 3.8 or higher:</strong> The programming language used for this project. 🐍</li>
    <li><strong>PyTorch:</strong> The core deep learning library for model training and inference. 🔍</li>
    <li><strong>YOLOv5:</strong> The state-of-the-art object detection model. 📈</li>
    <li><strong>OpenCV:</strong> An open-source computer vision library for image processing. 📷</li>
    <li><strong>Matplotlib:</strong> A plotting library for visualizations. 📊</li>
    <li><strong>LabelImg:</strong> A graphical image annotation tool for labeling images. ✍️</li>
</ul>

<h2 id="installation">Installation</h2>
<p>You can install the required libraries using pip:</p>
<pre><code>pip install torch torchvision torchaudio
pip install opencv-python matplotlib</code></pre>
<p>To clone the YOLOv5 repository, run:</p>
<pre><code>git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt</code></pre>

<h2 id="dataset-preparation">Dataset Preparation</h2>
<p>Prepare a dataset consisting of images that you want the model to detect. Ensure that the images are organized into folders based on their respective classes:</p>
<pre><code>dataset/
├── images/
│   ├── train/
│   │   ├── dog/
│   │   ├── cat/
│   │   ├── tv/
│   │   └── ... (other classes)
│   └── val/
│       ├── dog/
│       ├── cat/
│       ├── tv/
│       └── ... (other classes)
└── labels/
    ├── train/
    └── val/
</code></pre>

<h2 id="image-capture-for-dataset">Image Capture for Dataset</h2>
<p>Use a webcam or camera to capture images for your dataset. Aim for a diverse range of images for each class to improve model accuracy. In this project, we will capture 10 images for two classes: "Awake" and "Drowsy." Below is the code for capturing these images:</p>
<pre><code>import cv2
import os
import time
import uuid

# Define labels and number of images per label
labels = ['awake', 'drowsy']
number_imgs = 5

# Path where images will be saved
IMAGES_PATH = r'D:/PYTHON PROJECT/object-detection/yolov5/data/images'

# Make sure the directory exists
os.makedirs(IMAGES_PATH, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(2)  # 2 seconds delay before starting to capture images

    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num + 1))

        # Capture frame from webcam
        ret, frame = cap.read()

        if ret:
            # Naming out image path
            imgname = os.path.join(IMAGES_PATH, label + '.' + str(uuid.uuid1()) + '.jpg')

            # Write image to file
            cv2.imwrite(imgname, frame)
            print(f'Saved: {imgname}')

            # Display the frame for preview
            cv2.imshow('Image Collection', frame)

            # 2-second delay between captures
            time.sleep(2)

            # Stop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()</code></pre>
<p>This code initializes the webcam, captures 5 images for the "Awake" class, and then captures 5 images for the "Drowsy" class. Make sure to create the corresponding directories before running the code. 📸</p>

<h2 id="labeling-with-labelimg">Labeling with LabelImg</h2>
<p>After capturing images, use <strong>LabelImg</strong> to annotate the objects in your images. This will create the necessary bounding box labels for training the model.</p>
<pre><code>python labelImg.py</code></pre>
<p>After running this command, follow the UI instructions to annotate your images. 🖼️</p>

<h2 id="training-the-model">Training the Model</h2>
<p>Once your dataset is prepared and labeled, you can begin training the YOLOv5 model. Use the following command to start training:</p>
<pre><code>python train.py --img 640 --batch 16 --epochs 50 --data your_dataset.yaml --weights yolov5s.pt</code></pre>

<h2 id="inference">Inference</h2>
<p>After training, use the following command to run inference on new images:</p>
<pre><code>python detect.py --source your_image.jpg --weights runs/train/exp/weights/best.pt --img 640</code></pre>
<p>This will output images with detected objects and their respective labels. 🎯</p>

<h2>Tools and Technologies</h2>
<ul>
    <li><strong>YOLOv5:</strong> The object detection model used in this project. 🛠️</li>
    <li><strong>Python:</strong> The programming language used for development. 🐍</li>
    <li><strong>OpenCV:</strong> For image processing tasks. 📷</li>
    <li><strong>LabelImg:</strong> For image annotation. ✍️</li>
</ul>

<h2>How to Use</h2>
<ol>
    <li>Clone this repository. 📥</li>
    <li>Prepare your dataset and label your images using LabelImg. 🖼️</li>
    <li>Train the YOLOv5 model on your dataset. 🚀</li>
    <li>Run inference on new images to detect objects. 👀</li>
    <li>Explore and modify the code to suit your needs! 🔍</li>
</ol>

<h2>Conclusion</h2>
<p>This project demonstrates the effective application of YOLOv5 for detecting various objects. By capturing images of different states like "Awake" and "Drowsy," we showcase the model's ability to recognize diverse classes. The project highlights the significance of dataset preparation, image annotation, and model training in achieving high accuracy. With ongoing advancements in AI, this work serves as a stepping stone for innovative applications in computer vision! 🌟</p>
