from imutils import paths
import itertools
import random
import imutils
import time 
import cv2 

# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files("models/")
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each of the model paths, then combine the two lists together
models = list(zip(range(0, len(modelPaths)), modelPaths))

# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# read random imae from images folder
imageTitle = random.choice(list(paths.list_files("images/")))
print("[INFO] working image {}".format(imageTitle.split("/")[-1]))
image = cv2.imread(imageTitle)
image = imutils.resize(image, width=600)

while True:
    origin = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the frame, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    print("[INFO] neural style transfer took {:.4f} seconds".format(
	                            end - start))

    # show the original frame along with the output neural style transfer
    cv2.imshow("Input", image)
    cv2.imshow("Output", output)
    key = cv2.waitKey(0) 

    # if the `n` key is pressed (for "next"), load the next neural style transfer model
    if key == ord("n"):
        # grab the next neural style transfer model model and load it
        (modelID, modelPath) = next(modelIter)
        print("[INFO] {}. {}".format(modelID + 1, modelPath))
        net = cv2.dnn.readNetFromTorch(modelPath)
    # otheriwse, if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
# do a bit of cleanup
