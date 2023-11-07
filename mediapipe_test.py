import cv2
import mediapipe as mp

# Initialize MediaPipe Objectron.
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Set up the Objectron.
objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99,
                                   model_name='Shoe')  # You can set 'Cup', 'Chair', 'Camera' or 'Shoe'

# Initialize the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = objectron.process(image)

    # Draw the object detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

    # Display the resulting frame
    cv2.imshow('MediaPipe Objectron', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
