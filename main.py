import cv2
import AVFoundation as AV

def find_camera_index(name):
    # OpenCV on macOS sorts cameras by uniqueID, so the iPhone (Continuity Camera) can land on index 0
    devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + \
        list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    devices.sort(key=lambda d: d.uniqueID())
    for i, d in enumerate(devices):
        if name in d.localizedName():
            return i
    raise RuntimeError(f"Camera '{name}' not found")


cam = cv2.VideoCapture(find_camera_index("MacBook"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

while True:
    success, frame = cam.read()
    
    if not success:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 4)
    flipped = cv2.flip(frame, 1)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(flipped, (x,y), (x+w, y+h), (0,255,0),2)
        
    cv2.imshow('Real time stream:', flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



# ============================================================================

# def find_camera_index(name):
#     # OpenCV on macOS sorts cameras by uniqueID, so the iPhone (Continuity Camera) can land on index 0
#     devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + \
#         list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
#     devices.sort(key=lambda d: d.uniqueID())
#     for i, d in enumerate(devices):
#         if name in d.localizedName():
#             return i
#     raise RuntimeError(f"Camera '{name}' not found")


# cam = cv2.VideoCapture(find_camera_index("MacBook"))
# # cam = cv2.VideoCapture(0)

# while True:
#     success, frame = cam.read() # => True ,False
    
#     if not success:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(frame, (15,15),0)
    
#     edges1 = cv2.Canny(frame, 100, 200)
#     edges2 = cv2.Canny(frame, 50, 250)
    
#     flipped = cv2.flip(frame, 1)
    
#     cv2.imshow("Origincal" ,frame)
#     # cv2.imshow("Gray" ,gray)
#     # cv2.imshow("Blurred" ,blurred)
#     # cv2.imshow("Edges 1", edges1)
#     # cv2.imshow("Edges 2", edges2)
#     cv2.imshow("Flipped", flipped)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cam.release()
# cv2.destroyAllWindows()