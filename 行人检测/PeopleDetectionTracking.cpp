#include "PeopleDetectionTracking.h"
extern double accSpeed;
void peopleDetectionTracking(const string videoPath,const string outputPath)
{


	VideoCapture capVideo;

	capVideo.open(videoPath);

	if (!capVideo.isOpened()) 
	{
		cout << "错误：没有读取到视频文件！" << endl;
		return;
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2)
	{
		cout << "错误：视频帧数小于2！" << endl;
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


	Mat imgFrame1;
	Mat imgFrame2;
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	//定义计数横线位置
	Point crossingLine[2];
	int intHorizontalLinePosition = (int)round((double)imgFrame1.rows * 0.35);
	crossingLine[0].x = imgFrame1.cols * 0.276;
	crossingLine[0].y = imgFrame1.rows * 0.556;
	crossingLine[1].x = imgFrame1.cols * 0.674;
	crossingLine[1].y = imgFrame1.rows * 0.556;

	char chCheckForEscKey = 0;
	bool blnFirstFrame = true;
	int frameCount = 2;
	int peopleCount = 0;
	int peopleDCount = 0;
   
	vector<Blob> blobs;
	int out = 0;//每30帧输出一次这段时间内通过的车数
	ofstream outfile("./cache/out-people.txt");

	//定义文件输出流 
	ofstream oFile;

	//打开要输出的文件 
	oFile.open("./cache/行人信息摘要.csv", ios::out | ios::trunc);    // 这样就很容易的输出一个需要的excel 文件
	oFile << "当前视频时间" << "," << "累计通过行人总数" << "," << "1s内通过行人总数" << "," << "行人通过平均速率" << endl;
	
	while (capVideo.isOpened() && chCheckForEscKey != 27)
	{
		

		//获取当前帧的距离视频开始的时间位置（ms）
		//frameTime.push_back(capVideo.get(CV_CAP_PROP_POS_MSEC));
		if (out++ == 30)
		{
			peopleDCount = peopleCount - peopleDCount;
			string currentTime = to_string(capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000);
			string subCurrentTime = currentTime.substr(0, currentTime.size() - 5) + "秒";
			string avgSpeed;
			if (peopleDCount != 0)
				avgSpeed = to_string(accSpeed / peopleDCount);
			else
				avgSpeed = to_string(0);
			string subAvgSpeed = avgSpeed.substr(0, avgSpeed.size() - 5) + "km/h";

			outfile << "REAL-TIME：";
			outfile << subCurrentTime;
			outfile << "    ";
			outfile << "TOTAL-PEOPLE：";
			outfile << peopleCount;
			outfile << "    ";
			outfile << "REAL-TIME: ";
			outfile << capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000;
			outfile << "    ";
			outfile << "REAL-PEOPLE: ";
			outfile << peopleDCount << endl;
			//结合当前时间，写出到流量报告
			oFile << subCurrentTime << "," << peopleCount << "," << peopleDCount << "," << subAvgSpeed << endl;
			peopleDCount = peopleCount;
			accSpeed = 0;
			out = 0;
		}

		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();
		Mat imgDifference;
		Mat imgThresh;

		//减除法背景建模并更新背景
		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);
		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		
		//获得二值化图像
		threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
		//imshow("imgThresh", imgThresh);


		//获取常用的结构元素的形状
		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement15x15 = getStructuringElement(MORPH_RECT, Size(15, 15));

		//闭运算排除小型黑洞
		for (unsigned int i = 0; i < 2; i++)
		{
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}

		Mat imgThreshCopy = imgThresh.clone();


		//寻找并绘制轮廓，每个轮廓存储为一个点向量
		vector<vector<Point> > contours;
		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//drawAndShowContours(imgThresh.size(), contours, "imgContours");

		//寻找并绘制凸包
		vector<vector<Point> > convexHulls(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++)
			convexHull(contours[i], convexHulls[i]);

		vector<Blob> currentFrameBlobs; //当前帧中的团块

		//drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
		//判断团块是否为行人
		for (auto &convexHull : convexHulls)
		{
			Blob possibleBlob(convexHull);
			possibleBlob.judgePeople(possibleBlob);
			if (possibleBlob.blnIsPeople)
				currentFrameBlobs.push_back(possibleBlob);
		}
		//drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		//将当前行人团块放入总车辆团块集合
		if (blnFirstFrame == true)
			for (auto &currentFrameBlob : currentFrameBlobs)
				blobs.push_back(currentFrameBlob);
		else
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		//drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

		// get another copy of frame 2 since we changed the previous frame 2 copy in the processing above
		imgFrame2Copy = imgFrame2.clone();
		drawBlobInfoOnImage(blobs, imgFrame2Copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine3(blobs, intHorizontalLinePosition, peopleCount);
		if (blnAtLeastOneBlobCrossedTheLine == true)
			line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
		else
			line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		drawPeopleCountOnImage(peopleCount, imgFrame2Copy);
		imshow("imgFrame2Copy", imgFrame2Copy);
		writeVideo << imgFrame2Copy;


		/*读取下一帧图像*/
		currentFrameBlobs.clear();
		imgFrame1 = imgFrame2.clone();
		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT))
		{
			capVideo.read(imgFrame2);
		}
		else
		{
			cout << "视频结束" << endl;
			break;
		}

		blnFirstFrame = false;
		frameCount++;
		chCheckForEscKey = waitKey(1);
	}
	outfile.close();
	if (chCheckForEscKey != 27)
		waitKey(1);
}