import cv2
import face_recognition
from PIL import Image, ImageDraw


def recognize_face_pic_gpt(image_path):
    # 加载图片到numpy数组
    image = face_recognition.load_image_file(image_path)

    # 找到所有人脸的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    # 转换图片格式
    pil_image = Image.fromarray(image)

    # 创建一个PIL绘图对象
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        # 打印这个找到的脸的面部特征
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # 让我们在图像中描绘出每个人脸特征！
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)

    # 展示图片
    pil_image.show()


def recognize_face_pic(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否正确读取
    if image is None:
        print("Error: Cannot read the image. Please check the file path.")
        return

    # 识别人脸
    face_locations = face_recognition.face_locations(image)

    # 绘制人脸矩形框
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # 显示图片
    cv2.imshow('Recognized Image', image)

    # 等待按键，然后关闭显示窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def recognize_face_video():

    # 获取摄像头的视频流
    video_capture = cv2.VideoCapture(0)

    while True:
        # 获取一帧视频
        ret, frame = video_capture.read()

        # 使用face_recognition库找到所有的人脸
        face_locations = face_recognition.face_locations(frame)

        # 为每一个人脸画一个矩形
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 显示带有人脸标记的视频帧
        cv2.imshow('Video', frame)

        # 如果按下q键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流并关闭窗口
    video_capture.release()
    cv2.destroyAllWindows()


def compare_pic_video(image_path):

    # 先为已知的人生成人脸编码
    known_person_image = face_recognition.load_image_file(image_path)
    known_person_face_encoding = face_recognition.face_encodings(known_person_image)[0]

    # 获取摄像头的视频流
    video_capture = cv2.VideoCapture(0)

    while True:
        # 获取一帧视频
        ret, frame = video_capture.read()

        # 找到视频帧中所有的人脸和人脸编码
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # # 对每个人脸进行识别
        # for face_encoding in face_encodings:
        #     # 检查人脸是否与已知人匹配
        #     match = face_recognition.compare_faces([known_person_face_encoding], face_encoding)
        #
        #     # 如果找到匹配，标记为已知人的名字
        #     name = "Unknown"
        #     if match[0]:
        #         name = "Known Person"
        #
        #     # 画出人脸位置，并标注名字
        #     top, right, bottom, left = face_locations[0]
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #     cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        # 对每个人脸进行识别
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 检查人脸是否与已知人匹配
            matches = face_recognition.compare_faces([known_person_face_encoding], face_encoding)

            name = "Unknown"  # 默认为 "Unknown"

            if True in matches:
                name = "Known Person"

            # 画出人脸位置，并标注名字
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        # 显示结果视频帧
        cv2.imshow('Video', frame)

        # 如果按下q键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流并关闭窗口
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # image_path = "img/rich_women.jpg"
    # create_and_show_mirror_image(image_path)
    # recognize_face_pic(image_path)
    image_path = "img/me.jpg"
    # recognize_face_pic_gpt(image_path)
    # recognize_face_video()
    compare_pic_video(image_path)
