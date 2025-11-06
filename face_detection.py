import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ask user for image path
image_path = input("Enter full image path: ").strip()

# Read the image
image = cv2.imread(image_path)

# Validate image
if image is None:
    print("Error: Could not read image. Please check the path and try again.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles and labels
for idx, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(image, f"Face {idx}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display total count on image
cv2.putText(image, f"Total Faces Detected: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Show image window
cv2.imshow("Face Detection", image)

# Wait for 5 seconds OR a key press, then close automatically
cv2.waitKey(5000)  # 5000 ms = 5 seconds
cv2.destroyAllWindows()

# Print result in terminal after closing image

print(f"Total Faces Detected: {len(faces)}")

print("Execution was successfull.")

cv2.imshow("Face Detection", image)
cv2.waitKey(0)  # waits indefinitely until a key is pressed
cv2.destroyAllWindows()
