import cv2
import numpy as np

def count_objects(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Check the path!")
        return
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur (to remove noise)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 4: Thresholding - Convert to pure black & white
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Find contours (edges of each object)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw contours on the original image
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    # Step 7: Count and display the number of objects
    object_count = len(contours)
    print(f"✅ Objects detected: {object_count}")

    # Step 8: Display the count on top of the image
    cv2.putText(output, f'Total Objects: {object_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Step 9: Show the image
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image with Contours", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 🔹 MAIN PROGRAM EXECUTION
if __name__ == "__main__":
    path = input("Enter the path of the image file: ").strip()
    count_objects(path)
