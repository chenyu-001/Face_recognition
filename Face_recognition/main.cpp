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
	cv::Mat image; // ����һ����ͼ��
	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;
	image = cv::imread(strPath); // ��ȡ����ͼ��
	if (image.empty()) { // ������
		// δ����ͼ�񡭡�
		// ������ʾһ��������Ϣ
		// ���˳�����
		std::cout << "no image!\n";
	}
	else
	{
		// ���崰�ڣ���ѡ��
		cv::namedWindow("Original Image");
		// ��ʾͼ��
		cv::imshow("Original Image", image);
		cv::waitKey(0); // 0 ��ʾ��Զ�صȴ�����
		// �����������ʾ�ȴ��ĺ�����
		cv::Mat result; // ������һ���յ�ͼ��
		cv::flip(image, result, 1); // ������ʾˮƽ
		// 0 ��ʾ��ֱ
		// ������ʾˮƽ�ʹ�ֱ
		cv::namedWindow("Output Image"); // �������
		cv::imshow("Output Image", result);
		cv::waitKey(0); // 0 ��ʾ��Զ�صȴ�����
		// �����������ʾ�ȴ��ĺ�����
		cv::imwrite("output.bmp", result); // ������
	}
	system("pause");
}

void eg_face(string strPath) {
	Mat img = imread(strPath);
	namedWindow("display", 0);
	imshow("display", img);
	/*********************************** 1.�������������  ******************************/
	// ��������������
	CascadeClassifier cascade;
	// ����ѵ���õ� �����������.xml��
	//ע��·�����⣬��ǰĿ¼����һ��Ŀ¼�е�xml�ļ�����
	const string path = "../xml/haarcascade_frontalface_alt2.xml";
	if (!cascade.load(path))
	{
		cout << "cascade load failed!\n";
	}
	//��ʱ
	double t = 0;
	t = (double)getTickCount();
	/*********************************** 2.������� ******************************/
	vector<Rect> faces(0);
	cascade.detectMultiScale(img, faces, 1.1, 2, 0, Size(100, 100));
	cout << "detect face number is :" << faces.size() << endl;
	/********************************  3.��ʾ�������ο� ******************************/
	if (faces.size() > 0)
	{
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(img, faces[i], Scalar(150, 0, 0), 3, 8, 0);
		}
	}
	else cout << "δ��⵽����" << endl;
	t = (double)getTickCount() - t;  //getTickCount():  Returns the number of ticks per second.
	cout << "���������ʱ��" << t * 1000 / getTickFrequency() << "ms (���������ģ�͵�ʱ�䣩" << endl;
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
