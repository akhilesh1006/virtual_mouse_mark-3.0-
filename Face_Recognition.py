import cv2
import face_recognition
import numpy as np

# Load reference image and extract face encoding (embedding)
ref_img_path = "ref.jpg"
ref_image = cv2.imread(ref_img_path)
ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)  # Convert to RGB (face_recognition uses RGB)
ref_face_encoding = face_recognition.face_encodings(ref_image_rgb)

# Check if we found a face in the reference image
if len(ref_face_encoding) == 0:
    print("No face detected in the reference image.")
    exit()
else:
    ref_face_encoding = ref_face_encoding[0]  # Only one face expected

print("Reference image face encoding extracted.")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to compare two embeddings
def compare_embeddings(embedding1, embedding2, tolerance=0.6):
    # Compute Euclidean distance between two face embeddings
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    return distance < tolerance  # Return True if the distance is within the tolerance

frame_count = 0  # To process every nth frame

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    if frame_count % 5 != 0:  # Process every 5th frame
        continue

    # Resize the frame to reduce processing time (optional, tweak for performance)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the reference encoding
        match = compare_embeddings(ref_face_encoding, face_encoding)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)  # Rescale coordinates
        
        # Display match or no match
        if match:
            cv2.putText(frame, "Match", (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with bounding boxes and match info
    cv2.imshow("Webcam Feed", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
