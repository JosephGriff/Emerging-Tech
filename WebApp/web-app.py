## Author = Joseph  Griffith
## Student No. = G00350112

## Web Application Using Flask

# flask,tensor and numpy imports
import flask as fl
import tensorflow as tf

# Used to plot data
import numpy as np

# Used for encoding and decoding data
import base64

# Library of python bindings for visual problems
import cv2

# Imports from Python Image Library
from PIL import Image, ImageOps

#Used to import model in notebook
from keras.models import load_model

# Using load model import to bring in model created in notebook
model = load_model('../Digit_Predicter.h5')

# Creating the web application
app = fl.Flask(__name__)

# Resizing vars of the mnist dataset
height = 28
width = 28
size = height, width

## had: Tensor Tensor("dense_6/Softmax:0", shape=(?, 10), dtype=float32) is not an element of this graph. - error so adding in default graph to fix
# had to add graph = tf.get_default_graph() then global graph into the post 
## Used https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1 to solve
graph = tf.get_default_graph()


# Routing to web-app.html
# Route for home page
@app.route('/')
def home():
    # returns the static html file
    return fl.render_template('web-app.html')

#image is sent to the predict route where the prediction occurs
@app.route('/predict', methods=['POST'])
def convertImage():
    global graph
    with graph.as_default():
        # Fetches the image from the request
        encoded = fl.request.values[('imgBase64')]

        #Decoded decodes the dataURL while removing the extra 22 from the start of the image array index
        decoded = base64.b64decode(encoded[22:])

        # save the image
        with open('image.png', 'wb') as f:
            f.write(decoded)
        
        # Open the recently created image
        # Using the ImageOps import from Python Image Library(PIL) Resixe the image to fit the mnist dataset
        userImage = Image.open("image.png")
        newImage = ImageOps.fit(userImage, size, Image.ANTIALIAS)

        # save the newly resized image
        newImage.save("imageResized.png")

        # cv2 loads the new images
        cv2Image = cv2.imread("imageResized.png")

        # Converting the new image to grayscale, reshaping and adding to nparray
        grayScaleImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
        # Converting to float32 and dividing by 255 for attempted normilization(Does not really impact accuracy of web app)
        grayScaleArray = np.array(grayScaleImage, dtype=np.float32).reshape(1, 784)
        grayScaleArray /= 255

        # setter and getter to return the predicition from the model
        setPrediction = model.predict(grayScaleArray)
        getPrediction = np.array(setPrediction[0])

        # np.argmax returns the highest value ie what should be the same as the digit passed
        predictedNumber = str(np.argmax(getPrediction))
        print(predictedNumber)

        # returns the predicted number
        return predictedNumber

# runs the app
app.run(threaded=False)
