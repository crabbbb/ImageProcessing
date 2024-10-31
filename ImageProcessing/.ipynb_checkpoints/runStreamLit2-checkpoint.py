import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf


color_model = tf.keras.models.load_model('./color_segmentation_apple.h5')
edge_model = tf.keras.models.load_model('./edge_canny_apple.h5')
texture_model = tf.keras.models.load_model('./texture_apple.h5')

def preprocess_img(img):
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    return images

# Function for color-based segmentation
def color_segmentation(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (256,256))
    
    # Define lower and upper bounds for red color
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])
    
    # Threshold the HSV image to get a binary mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to clean up the mask (optional)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Function for Texture Analysis
def compute_lbp(image, radius=1, neighbors=8):
    # Convert the image to grayscale if it's in color
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute the LBP image
    lbp = np.zeros_like(gray)
    for i in range(len(gray)):
        for j in range(len(gray[0])):
            center = gray[i, j]
            pattern = 0
            for k in range(neighbors):
                x = i + int(round(radius * np.cos(2 * np.pi * k / neighbors)))
                y = j - int(round(radius * np.sin(2 * np.pi * k / neighbors)))
                if x >= 0 and x < len(gray) and y >= 0 and y < len(gray[0]):
                    pattern |= (gray[x, y] >= center) << k
            lbp[i, j] = pattern

    return lbp

# Function for edge detection
def edge_detection(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    return edges

def main():
    st.title('Image Viewer and Camera App')

    # Input field for typing image file path
    image_path = st.text_input('Enter Image File Path:')
    
    # Button to display the image
    if st.button('Color Segmentation Image'):
        if image_path:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            color_segmented = color_segmentation(img)
            images = preprocess_img(color_segmented)
            prediction = color_model.predict(images)
            # print(prediction)
            # print(prediction[0])
            # print(prediction[0][0])
            if prediction[0][0] == 0:
                st.text("ColorSegmentation: This is an apple")
            else:
                st.text("ColorSegmentation: This is a defect apple")

            cv2.imshow("image", color_segmented)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if st.button('Texture Analysis Image'):
        if image_path:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256,256))
            lbp = compute_lbp(img)
            images = preprocess_img(lbp)
            prediction = texture_model.predict(images)
            # print(prediction)
            # print(prediction[0])
            # print(prediction[0][0])
            if prediction[0][0] == 0:
                st.text("TextureMode: This is an apple")
            else:
                st.text("TextureMode: This is a defect apple")
    
            lbp = cv2.resize(lbp, (256,256))
            cv2.imshow("image", lbp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if st.button('Edge Detection Image'):
        if image_path:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256,256))
            edge = edge_detection(img)
            images = preprocess_img(edge)
            prediction = edge_model.predict(images)
            # print(prediction)
            # print(prediction[0])
            # print(prediction[0][0])
            if prediction[0][0] == 0:
                st.text("EdgeModel: This is an apple")
            else:
                st.text("EdgeModel: This is a defect apple")
            
            edge = cv2.resize(edge, (256,256))
            cv2.imshow("image", edge)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # Button to open camera with Color Segmentation
    if st.button('Camera With Color Segmentation'):
        cap = cv2.VideoCapture(0)
        apples_cascade = cv2.CascadeClassifier('./cascade_apples.xml')

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            apples = apples_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in apples:
                # Calculate padding values to make the bounding box (256, 256)
                pad_x = max(0, (256 - w) // 2)
                pad_y = max(0, (256 - h) // 2)
                # Adjust the coordinates of the bounding box
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
                # Draw the adjusted bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                
                try:
                    # Crop the area including the background
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, (256, 256))

                    color_segmented = color_segmentation(cropped_img)
                    # img = image.array_to_img(color_segmented, scale=False)
                    # Process the cropped image as needed
                    images = preprocess_img(color_segmented)

                    val = color_model.predict(images)
                    text = ""
                    if val == 0:
                        text = "Apple"
                    else:
                        text = "Defect Apple"
                        
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except Exception as e:
                    print("Exception:", e)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Button to open camera with Texture Analysis
    if st.button('Camera With Texture Analysis'):
        cap = cv2.VideoCapture(0)
        apples_cascade = cv2.CascadeClassifier('./cascade_apples.xml')

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            apples = apples_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in apples:
                # Calculate padding values to make the bounding box (256, 256)
                pad_x = max(0, (256 - w) // 2)
                pad_y = max(0, (256 - h) // 2)
                # Adjust the coordinates of the bounding box
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
                # Draw the adjusted bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                
                try:
                    # Crop the area including the background
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, (256, 256))

                    lbp = compute_lbp(cropped_img)
                    # img = image.array_to_img(color_segmented, scale=False)
                    # Process the cropped image as needed
                    images = preprocess_img(lbp)

                    val = color_model.predict(images)
                    text = ""
                    if val == 0:
                        text = "Apple"
                    else:
                        text = "Defect Apple"
                        
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except Exception as e:
                    print("Exception:", e)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if st.button('Camera With Edge Detection'):
        cap = cv2.VideoCapture(0)
        apples_cascade = cv2.CascadeClassifier('./cascade_apples.xml')

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            apples = apples_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in apples:
                # Calculate padding values to make the bounding box (256, 256)
                pad_x = max(0, (256 - w) // 2)
                pad_y = max(0, (256 - h) // 2)
                # Adjust the coordinates of the bounding box
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
                # Draw the adjusted bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                
                try:
                    # Crop the area including the background
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, (256, 256))

                    edge = edge_detection(cropped_img)
                    # img = image.array_to_img(color_segmented, scale=False)
                    # Process the cropped image as needed
                    images = preprocess_img(edge)

                    val = color_model.predict(images)
                    text = ""
                    if val == 0:
                        text = "Apple"
                    else:
                        text = "Defect Apple"
                        
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except Exception as e:
                    print("Exception:", e)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()