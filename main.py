import cv2 as cv
import mediapipe as mp 

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a MediaPipe Hands object
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# OpenCV capture for webcam
cap = cv.VideoCapture(0)
cap.set(3, 2120)  # Set a smaller width to reduce the load
cap.set(4, 1080)  # Set a smaller height to reduce the load

# Define the virtual keyboard layout
keys = [["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P", "^", "$"],
        ["Q", "S", "D", "F", "G", "H", "J", "K", "L", "M", "%", "*"],
        ["W", "X", "C", "V", "B", "N", ",", ";", ":", "!", ".", "?"]]

finalText = ""  # String to store the typed text
clicked = False  # To track if a key is clicked

# Define a class for keys in the virtual keyboard
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

# Generate buttons for each key in the virtual keyboard
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Add special keys (Space and Delete)
buttonList.append(Button([50, 350], "Space", [885, 85]))
buttonList.append(Button([950, 350], "Delete", [285, 85]))

# Function to draw all buttons (keyboard)
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        # Draw the button (rectangle) with text
        if button.text == "Space" or button.text == "Delete":
            cv.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), cv.FILLED)
            text_x = x + int(w * 0.35) - 50
            text_y = y + int(h * 0.65)
            cv.putText(img, button.text, (text_x, text_y), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        else:
            cv.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), cv.FILLED)
            cv.putText(img, button.text, (x + 25, y + 60), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Function to detect hand landmarks and return a list
def handLandmarks(colorImg):
    landmarkList = []
    rgb_img = cv.cvtColor(colorImg, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for index, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = colorImg.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([index, x, y])
                # Optionally draw a circle on each landmark
                cv.circle(colorImg, (x, y), 5, (0, 255, 0), -1)  # Show hand landmarks

            # Draw hand landmarks and connections for better visualization
            mp_drawing.draw_landmarks(colorImg, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return landmarkList

# Main loop
while True:
    res, img = cap.read()

    if not res:
        break

    img = cv.flip(img, 1)  # Flip image for mirror effect
    lmlist = handLandmarks(img)  # Get hand landmarks
    img = drawAll(img, buttonList)  # Draw the virtual keyboard

    if lmlist:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            # Check if the index finger is over a key
            if x < lmlist[8][1] < x + w and y < lmlist[8][2] < y + h:
                # Highlight the key if the index finger is over it
                cv.rectangle(img, button.pos, (x + w, y + h), (128, 128, 128), cv.FILLED)
                cv.putText(img, button.text, (x + 25, y + 60), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # Check if the index and middle fingers are close together (click)
                if lmlist[8][2] < lmlist[7][2] and lmlist[12][2] < lmlist[11][2] and not clicked:
                    if button.text == "Space":
                        finalText += " "
                    elif button.text == "Delete":
                        finalText = finalText[:-1]
                    else:
                        finalText += button.text
                    clicked = True

            # Reset click status when the fingers move apart
            if lmlist[8][2] > lmlist[7][2] or lmlist[12][2] > lmlist[11][2]:
                clicked = False

    # Display the typed text
    cv.rectangle(img, (50, 580), (1235, 680), (64, 64, 64), cv.FILLED)
    cv.putText(img, finalText, (60, 645), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    # Show the image with the keyboard and the hand landmarks
    cv.imshow("Virtual Keyboard", img)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv.destroyAllWindows()
