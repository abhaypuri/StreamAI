import numpy as np
import imutils
import time
import cv2
import streamlit as st
import matplotlib.pyplot as plt

def semantic_segmentation():
    st.title("Semantic Segmentation")
    st.write("Semantic Segmentation using OpenCV. Can detect various aspects adn parts of an cityscape image using enet.")
    # load the class label names
    CLASSES = open("enet-cityscapes/enet-classes.txt").read().strip().split("\n")
    COLORS = open("enet-cityscapes/enet-colors.txt").read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    # initialize the legend visualization
    legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
        # draw the class name + color on the legend
        color = [int(c) for c in color]
        cv2.putText(legend, className, (5, (i * 25) + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
            tuple(color), -1)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNet("enet-cityscapes/enet-model.net")
    # load the input image, resize it, and construct a blob from it,
    # but keeping mind mind that the original input image dimensions
    # ENet was trained on was 1024x512
    image = cv2.imread("images/cityscapes4.jpg")
    image = imutils.resize(image, width=512)
    image_display = image[:, :, [2, 1, 0]] # BGR -> RGB

    st.markdown("*Original image:*")
    plt.figure(figsize=(12,8))
    plt.imshow(image_display)
    st.pyplot()

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
        swapRB=True, crop=False)
    # perform a forward pass using the segmentation model
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()
    # show the amount of time inference took
    print("[INFO] inference took {:.4f} seconds".format(end - start))
    # infer the total number of classes along with the spatial dimensions
    # of the mask image via the shape of the output array
    (numClasses, height, width) = output.shape[1:4]
    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(output[0], axis=0)
    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    mask = COLORS[classMap]
    # resize the mask and class map such that its dimensions match the
    # original size of the input image (we're not using the class map
    # here for anything else but this is how you would resize it just in
    # case you wanted to extract specific pixels/classes)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    # perform a weighted combination of the input image with the mask to
    # form an output visualization
    output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
    output_display = output[:, :, [2, 1, 0]] # BGR -> RGB
    legend_display = legend[:, :, [2, 1, 0]] # BGR -> RGB
    # show the input and output images
    st.markdown("*Segmented Image:*")
    plt.figure(figsize=(12,8))
    plt.imshow(output_display)
    st.pyplot()

    st.markdown("*Legend:*")
    plt.figure(figsize=(12,8))
    plt.imshow(legend_display)
    st.pyplot()
    # cv2.imshow("Legend", legend)
    # cv2.imshow("Input", image)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)

# semantic_segmentation()