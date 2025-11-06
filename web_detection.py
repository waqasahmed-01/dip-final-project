import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if webcam is accessible
if not cap.isOpened():
    print("Error: Could not access webcam. Please check your camera connection.")
    exit()

print("\n🎥 Webcam started — press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert each frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles and labels for each detected face
    for idx, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, f"Face {idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display total number of faces detected
    cv2.putText(frame, f"Total Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow("Webcam Face Detection", frame)

    # Press 'q' to quit webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n Webcam stopped. Total faces detected in last frame: {len(faces)}\n")
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
