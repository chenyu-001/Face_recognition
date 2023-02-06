#include <opencv2/core.hpp> 
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

using namespace std;

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"

using namespace cv;
using namespace cv::ml;

void first(string strPath) {
	cv::Mat image; // 创建一个空图像
	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;
	image = cv::imread(strPath); // 读取输入图像
	if (image.empty()) { // 错误处理
		// 未创建图像……
		// 可能显示一个错误消息
		// 并退出程序
		std::cout << "no image!\n";
	}
	else
	{
		// 定义窗口（可选）
		cv::namedWindow("Original Image");
		// 显示图像
		cv::imshow("Original Image", image);
		cv::waitKey(0); // 0 表示永远地等待按键
		// 键入的正数表示等待的毫秒数
		cv::Mat result; // 创建另一个空的图像
		cv::flip(image, result, 1); // 正数表示水平
		// 0 表示垂直
		// 负数表示水平和垂直
		cv::namedWindow("Output Image"); // 输出窗口
		cv::imshow("Output Image", result);
		cv::waitKey(0); // 0 表示永远地等待按键
		// 键入的正数表示等待的毫秒数
		cv::imwrite("output.bmp", result); // 保存结果
	}
	system("pause");
}

void eg_face(string strPath) {
	Mat img = imread(strPath);
	namedWindow("display", 0);
	imshow("display", img);
	/*********************************** 1.加载人脸检测器  ******************************/
	// 建立级联分类器
	CascadeClassifier cascade;
	// 加载训练好的 人脸检测器（.xml）
	//注意路径问题，当前目录的上一个目录中的xml文件夹下
	const string path = "../xml/haarcascade_frontalface_alt2.xml";
	if (!cascade.load(path))
	{
		cout << "cascade load failed!\n";
	}
	//计时
	double t = 0;
	t = (double)getTickCount();
	/*********************************** 2.人脸检测 ******************************/
	vector<Rect> faces(0);
	cascade.detectMultiScale(img, faces, 1.1, 2, 0, Size(100, 100));
	cout << "detect face number is :" << faces.size() << endl;
	/********************************  3.显示人脸矩形框 ******************************/
	if (faces.size() > 0)
	{
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(img, faces[i], Scalar(150, 0, 0), 3, 8, 0);
		}
	}
	else cout << "未检测到人脸" << endl;
	t = (double)getTickCount() - t;  //getTickCount():  Returns the number of ticks per second.
	cout << "检测人脸用时：" << t * 1000 / getTickFrequency() << "ms (不计算加载模型的时间）" << endl;
	namedWindow("face_detect", 0);
	imshow("face_detect", img);
	waitKey(0);

	destroyWindow("display");
	destroyWindow("face_detect");
}
int main()
{
	string strPath = "../img/rich_women.jpg";
	//first(strPath);
	eg_face(strPath);
	return 0;
}
