import cv2
from mtcnn.mtcnn import MTCNN
from imutils import build_montages
from imutils import paths
detector = MTCNN()

img = cv2.imread("input & output/crowd.jpg")
location = detector.detect_faces(img)
count = 1
location_new=[]
for loc in location:
    if loc['confidence']>=0.99:
        location_new.append(loc)
print(type(location_new))
location_new.sort(key=lambda item: item.get("confidence"), reverse=True)
print(location_new)

if len(location_new) > 0:
# if location['confidence']>=0.99:
    for face in location_new:
        try:
            x, y, width, height = face['box']
            # print(x,y,width,height)
            x2, y2 = x + width, y + height
            # print(x2,y2)
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)
            sub_face = img[y:y + height, x:x + width]
            # cv2.imshow("crop_image", sub_face)
            # cv2.waitKey(0)
            FaceFileName = "cluster_faces/face_" + str(count) + ".jpg"
            count += 1
            cv2.imwrite(FaceFileName, sub_face)
        except Exception as e:
            print(e)

image_folder = './cluster_faces'
image_path = list(paths.list_images(image_folder))
imgs = []
for img_path in image_path:
    img = cv2.imread(img_path)
    imgs.append(img)

montages = build_montages(imgs, (75,75), (12,11))

for montage in montages:
    cv2.imshow("montage_output", montage)
    cv2.imwrite("output_montage_new.jpg", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("input & output/crowd_output.jpg",img)
# print("The Image was successfully saved")
