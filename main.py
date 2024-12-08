import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set dataset path
dataset_path = "DataUmbi"

# 1. Save Grayscale and RGB Data to Separate Sheets in Excel
def save_image_data_to_excel(image_path, excel_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (128, 128))
    resized_rgb = cv2.resize(img, (128, 128))

    # Split RGB channels
    red_channel = resized_rgb[:, :, 2]
    green_channel = resized_rgb[:, :, 1]
    blue_channel = resized_rgb[:, :, 0]

    # Convert to DataFrames
    gray_df = pd.DataFrame(resized_gray)
    red_df = pd.DataFrame(red_channel)
    green_df = pd.DataFrame(green_channel)
    blue_df = pd.DataFrame(blue_channel)

    # Write to Excel with separate sheets
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        gray_df.to_excel(writer, sheet_name="Grayscale", index=False)
        red_df.to_excel(writer, sheet_name="Red", index=False)
        green_df.to_excel(writer, sheet_name="Green", index=False)
        blue_df.to_excel(writer, sheet_name="Blue", index=False)

    print(f"Saved grayscale and RGB data to {excel_path}")

# 2. Preprocessing and Feature Extraction
def preprocess_and_extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))

    # Feature extraction
    texture_features = extract_texture_features(resized)
    shape_features = extract_shape_features(resized)

    # Add mean color features
    mean_red = np.mean(image[:, :, 2])
    mean_green = np.mean(image[:, :, 1])
    mean_blue = np.mean(image[:, :, 0])

    return np.hstack((texture_features, shape_features, mean_red, mean_green, mean_blue))

# 3. Extract Texture Features
def extract_texture_features(image):
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    return [contrast, energy, correlation, homogeneity, entropy]

# 4. Extract Shape Features
def extract_shape_features(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        moments = cv2.moments(cnt)
        metric = moments['m00'] / (moments['m10'] + moments['m01'] + 1e-5)
        major_axis = max(cnt[:, 0, 0]) - min(cnt[:, 0, 0])
        minor_axis = max(cnt[:, 0, 1]) - min(cnt[:, 0, 1])
        eccentricity = (1 - (minor_axis / (major_axis + 1e-5))) if major_axis > minor_axis else 0
        return [area, perimeter, metric, major_axis, minor_axis, eccentricity]
    return [0, 0, 0, 0, 0, 0]

# 5. Capture Image from Camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # Use default camera
    print("Press 's' to save the image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):  # Save image on pressing 's'
            image_path = "captured_image.png"
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            break
        elif key == ord('q'):  # Exit on pressing 'q'
            print("Exiting without saving.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return image_path if key == ord('s') else None

# Main Program
if __name__ == "__main__":
    # Load Dataset
    features, labels, filenames, label_map = load_dataset_with_labels_and_save(dataset_path, "Features.xlsx")

    # Split Data for Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train and Optimize Model
    knn_model = train_and_optimize_knn(X_train, y_train)

    # Evaluate Model
    y_pred = knn_model.predict(X_test)
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Predict Image from Camera
    image_path = capture_image_from_camera()
    if image_path:
        predicted_class = predict_new_image(knn_model, image_path, label_map)
        print(f"Predicted Class: {predicted_class}")
