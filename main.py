import cv2
import face_recognition


def load_data():
    # Load the known images
    image_of_person_1 = face_recognition.load_image_file("person_1.jpg")
    image_of_person_2 = face_recognition.load_image_file("person_2.jpg")

    # Get the face encoding of each person. This can fail if no one is found in the photo.
    person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
    person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

    # Create a list of known face encodings
    known_face_encodings = [
        person_1_face_encoding,
        person_2_face_encoding
    ]

    # Load the image we want to check
    unknown_image = face_recognition.load_image_file("unknown_person.jpg")

    # Get face encodings for any people in the picture
    face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Loop through each face found in the unknown image
    for unknown_face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
        if True in results:
            print("Access granted")
        else:
            print("Access denied")


def create_and_show_mirror_image(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否正确读取
    if image is None:
        print("Error: Cannot read the image. Please check the file path.")
        return

    # 创建镜像图片
    mirror_image = cv2.flip(image, 1)

    # 显示原始图片
    cv2.imshow('Original Image', image)

    # 显示镜像图片
    cv2.imshow('Mirror Image', mirror_image)

    # 等待按键，然后关闭显示窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "img/rich_women.jpg"
    create_and_show_mirror_image(image_path)
