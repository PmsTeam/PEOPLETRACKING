#include "DDetectionTracking.h"
extern double accSpeed;

void DDetectionTracking(const string sourcePath, const string outputPath)
{
	VideoCapture capVideo;

	capVideo.open(sourcePath);


	// 如果打开视频失败
	if (!capVideo.isOpened()) {
		//给出错误提示
		std::cout << "error reading video file" << std::endl << std::endl;
		// 如果不使用Windows，可能需要更改或删除该行
		//_getch();
		return ;
	}

	//capVideo.get(CV_CAP_PROP_FRAME_COUNT) 为视频文件中的帧数
	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		//如果视频的帧数小于2，给出错误信息
		std::cout << "error: video file must have at least two frames";
		// 如果不使用Windows，可能需要更改或删除该行
		//_getch();
		return;
	}


	//打开输出视频
	VideoWriter writeVideo;
	writeVideo.open(outputPath, // 输出视频文件名
		(int)capVideo.get(CV_FOURCC_PROMPT), // 也可设为CV_FOURCC_PROMPT，在运行时选取
		(double)capVideo.get(CV_CAP_PROP_FPS), // 视频帧率
		Size((int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT)), // 视频大小
		true); // 是否输出彩色视频
	ofstream oFile;//输出csv专用
	oFile.open("./cache//Car-People.csv", ios::out | ios::trunc);
	oFile << "当前视频时间" << "," << "累计通过车辆总数" << "," << "1s内通过车辆总数" << "," << "累计通过行人总数" << "," << "1s内通过行人总数" << endl;

	Mat imgFrame1;
	Mat imgFrame2;

	vector<Blob> blobs;

	Point crossingLine[2];
	Point crossingLine2[2];

	int carCount = 0;
	int peopleCount = 0;
	int carDCount = 0;
	int peopleDCount = 0;

	//同captuer >> imgFrame 将视频帧读出
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	//划线函数，用于流量统计用/////////////////////////////////////////
	int intHorizontalLinePosition = (int)std::round((double)imgFrame1.cols * 0.404);         //车
	crossingLine[0].x = imgFrame1.cols * 0.404;
	crossingLine[0].y = imgFrame1.rows * 0.250;
	crossingLine[1].x = imgFrame1.cols * 0.404;
	crossingLine[1].y = imgFrame1.rows * 0.572;

	int intHorizontalLinePosition2 = (int)std::round((double)imgFrame1.cols * 0.319);    //人
	crossingLine2[1].x = imgFrame1.cols * 0.319;
	crossingLine2[1].y = imgFrame1.rows * 0.628;
	crossingLine2[0].x = imgFrame1.cols * 0.319;
	crossingLine2[0].y = imgFrame1.rows * 0.842;
	///////////////////////////////////////////////////////////////////

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;

	int out = 0;//每30帧输出一次这段时间内通过的车数
	ofstream outfile("./cache/out-Double.txt");


	while (capVideo.isOpened() && chCheckForEscKey != 27) {     //capVideo.isOpende()判断视频读取或者摄像头调用是否成功，成功则返回true。


		if (out++ == 30)
		{
			peopleDCount = peopleCount - peopleDCount;
			carDCount = carCount - carDCount;

			string currentTime = to_string(capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000);
			string subCurrentTime = currentTime.substr(0, currentTime.size() - 5) + "秒";
			string avgSpeed;
			if (carDCount != 0)
				avgSpeed = to_string(accSpeed / carDCount);
			else
				avgSpeed = to_string(0);
			string subAvgSpeed = avgSpeed.substr(0, avgSpeed.size() - 5);
			oFile << subCurrentTime << "," << carCount << "," << carDCount << "," << peopleCount << "," << peopleCount << endl;

			outfile << "TOTAL-PEOPLE：";
			outfile << peopleCount;
			outfile << "    ";
			outfile << "REAL-TIME: ";
			outfile << capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000;
			outfile << "    ";
			outfile << "REAL-PEOPLE: ";
			outfile << peopleDCount << endl;
			peopleDCount = peopleCount;

			
			outfile << "TOTAL-CAR：";
			outfile << carCount;
			outfile << "    ";
			outfile << "REAL-TIME: ";
			outfile << capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000;
			outfile << "    ";
			outfile << "REAL-CAR: ";
			outfile << carDCount << endl;
			carDCount = carCount;
			out = 0;
		}



		//创建原始数据的副本
		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgDifference;
		Mat imgThresh;

		//颜色转换，CV_BGR2GRAY 将颜色从RGB向gray装换
		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		//高斯滤波函数，cv:Size()为高斯内核大小，0位在X,y方向的标准方差
		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		//计算前两个图像ROI数值的差值绝对值给第三个图像
		cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		//阈值门限分割cv::threshold，五个参数分别为 输入矩阵，输出矩阵，阈值，设置的最大值，阈值类型
		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		// cv::imshow("imgThresh", imgThresh);

		//OpenCV提供了一个函数getStructuringElement，可以获取常用的结构元素的形状：矩形（包括线形）MORPH_RECT、椭圆（包括圆形）MORPH_ELLIPSE及十字形MORPH_CROSS。
		Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

		//闭运算排除小型黑洞
		for (unsigned int i = 0; i < 2; i++)
		{
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}

		//克隆图像
		Mat imgThreshCopy = imgThresh.clone();

		vector<vector<Point> > contours;

		//函数cvFindContours从二值图像中检索轮廓，并返回检测到的轮廓的个数。
		//https://baike.baidu.com/item/cvFindContours/9560509?fr=aladdin
		//CV_RETR_EXTERNAL：只检索最外面的轮廓；CV_CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
		findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//显示轮廓
		drawAndShowContours(imgThresh.size(), contours, "imgContours");

		vector<vector<Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			cv::convexHull(contours[i], convexHulls[i]);         //寻找凸包
		}

		drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
		vector<Blob> currentFrameBlobs;

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);
			possibleBlob.DjudgePeople(possibleBlob);
			possibleBlob.DjudgeVehicle(possibleBlob);
			if (possibleBlob.blnIsPeople || possibleBlob.blnIsVehicle)
				currentFrameBlobs.push_back(possibleBlob);
			}

		drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true)
			for (auto &currentFrameBlob : currentFrameBlobs)
				blobs.push_back(currentFrameBlob);
		else
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);

		drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

		imgFrame2Copy = imgFrame2.clone();          //在上面的处理中，我们改变了之前的帧2拷贝，得到另一个框架2

		drawBlobInfoOnImage(blobs, imgFrame2Copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine2(blobs, intHorizontalLinePosition2, peopleCount);
		if (blnAtLeastOneBlobCrossedTheLine == true) {
			cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_BLACK, 2);
		}
		else {
			cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_RED, 2);
		}

		blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);
		if (blnAtLeastOneBlobCrossedTheLine == true) {
			cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
		}
		else {
			cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		}

		drawCarCountOnImage(carCount, imgFrame2Copy);
		drawPeopleCountOnImage(peopleCount, imgFrame2Copy);
		cv::imshow("imgFrame2Copy", imgFrame2Copy);
		writeVideo << imgFrame2Copy;

		currentFrameBlobs.clear();

		imgFrame1 = imgFrame2.clone();           //将第1帧移动到第2帧的位置

		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
			capVideo.read(imgFrame2);
		}
		else {
			std::cout << "end of video\n";
			break;
		}

		blnFirstFrame = false;
		frameCount++;
		chCheckForEscKey = cv::waitKey(1);
	}

	if (chCheckForEscKey != 27) {               //如果用户没有按下esc(即视频的结束)
		cv::waitKey(1);                         //让结束信息显示在窗口上
	}

}

