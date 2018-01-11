#include "DDetectionTracking.h"
extern double accSpeed;

void DDetectionTracking(const string sourcePath, const string outputPath)
{
	VideoCapture capVideo;

	capVideo.open(sourcePath);


	// �������Ƶʧ��
	if (!capVideo.isOpened()) {
		//����������ʾ
		std::cout << "error reading video file" << std::endl << std::endl;
		// �����ʹ��Windows��������Ҫ���Ļ�ɾ������
		//_getch();
		return ;
	}

	//capVideo.get(CV_CAP_PROP_FRAME_COUNT) Ϊ��Ƶ�ļ��е�֡��
	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		//�����Ƶ��֡��С��2������������Ϣ
		std::cout << "error: video file must have at least two frames";
		// �����ʹ��Windows��������Ҫ���Ļ�ɾ������
		//_getch();
		return;
	}


	//�������Ƶ
	VideoWriter writeVideo;
	writeVideo.open(outputPath, // �����Ƶ�ļ���
		(int)capVideo.get(CV_FOURCC_PROMPT), // Ҳ����ΪCV_FOURCC_PROMPT��������ʱѡȡ
		(double)capVideo.get(CV_CAP_PROP_FPS), // ��Ƶ֡��
		Size((int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT)), // ��Ƶ��С
		true); // �Ƿ������ɫ��Ƶ
	ofstream oFile;//���csvר��
	oFile.open("./cache//Car-People.csv", ios::out | ios::trunc);
	oFile << "��ǰ��Ƶʱ��" << "," << "�ۼ�ͨ����������" << "," << "1s��ͨ����������" << "," << "�ۼ�ͨ����������" << "," << "1s��ͨ����������" << endl;

	Mat imgFrame1;
	Mat imgFrame2;

	vector<Blob> blobs;

	Point crossingLine[2];
	Point crossingLine2[2];

	int carCount = 0;
	int peopleCount = 0;
	int carDCount = 0;
	int peopleDCount = 0;

	//ͬcaptuer >> imgFrame ����Ƶ֡����
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	//���ߺ�������������ͳ����/////////////////////////////////////////
	int intHorizontalLinePosition = (int)std::round((double)imgFrame1.cols * 0.404);         //��
	crossingLine[0].x = imgFrame1.cols * 0.404;
	crossingLine[0].y = imgFrame1.rows * 0.250;
	crossingLine[1].x = imgFrame1.cols * 0.404;
	crossingLine[1].y = imgFrame1.rows * 0.572;

	int intHorizontalLinePosition2 = (int)std::round((double)imgFrame1.cols * 0.319);    //��
	crossingLine2[1].x = imgFrame1.cols * 0.319;
	crossingLine2[1].y = imgFrame1.rows * 0.628;
	crossingLine2[0].x = imgFrame1.cols * 0.319;
	crossingLine2[0].y = imgFrame1.rows * 0.842;
	///////////////////////////////////////////////////////////////////

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;

	int out = 0;//ÿ30֡���һ�����ʱ����ͨ���ĳ���
	ofstream outfile("./cache/out-Double.txt");


	while (capVideo.isOpened() && chCheckForEscKey != 27) {     //capVideo.isOpende()�ж���Ƶ��ȡ��������ͷ�����Ƿ�ɹ����ɹ��򷵻�true��


		if (out++ == 30)
		{
			peopleDCount = peopleCount - peopleDCount;
			carDCount = carCount - carDCount;

			string currentTime = to_string(capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000);
			string subCurrentTime = currentTime.substr(0, currentTime.size() - 5) + "��";
			string avgSpeed;
			if (carDCount != 0)
				avgSpeed = to_string(accSpeed / carDCount);
			else
				avgSpeed = to_string(0);
			string subAvgSpeed = avgSpeed.substr(0, avgSpeed.size() - 5);
			oFile << subCurrentTime << "," << carCount << "," << carDCount << "," << peopleCount << "," << peopleCount << endl;

			outfile << "TOTAL-PEOPLE��";
			outfile << peopleCount;
			outfile << "    ";
			outfile << "REAL-TIME: ";
			outfile << capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000;
			outfile << "    ";
			outfile << "REAL-PEOPLE: ";
			outfile << peopleDCount << endl;
			peopleDCount = peopleCount;

			
			outfile << "TOTAL-CAR��";
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



		//����ԭʼ���ݵĸ���
		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgDifference;
		Mat imgThresh;

		//��ɫת����CV_BGR2GRAY ����ɫ��RGB��grayװ��
		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		//��˹�˲�������cv:Size()Ϊ��˹�ں˴�С��0λ��X,y����ı�׼����
		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		//����ǰ����ͼ��ROI��ֵ�Ĳ�ֵ����ֵ��������ͼ��
		cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		//��ֵ���޷ָ�cv::threshold����������ֱ�Ϊ ����������������ֵ�����õ����ֵ����ֵ����
		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		// cv::imshow("imgThresh", imgThresh);

		//OpenCV�ṩ��һ������getStructuringElement�����Ի�ȡ���õĽṹԪ�ص���״�����Σ��������Σ�MORPH_RECT����Բ������Բ�Σ�MORPH_ELLIPSE��ʮ����MORPH_CROSS��
		Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

		//�������ų�С�ͺڶ�
		for (unsigned int i = 0; i < 2; i++)
		{
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}

		//��¡ͼ��
		Mat imgThreshCopy = imgThresh.clone();

		vector<vector<Point> > contours;

		//����cvFindContours�Ӷ�ֵͼ���м��������������ؼ�⵽�������ĸ�����
		//https://baike.baidu.com/item/cvFindContours/9560509?fr=aladdin
		//CV_RETR_EXTERNAL��ֻ�����������������CV_CHAIN_APPROX_SIMPLE��ѹ��ˮƽ�ġ���ֱ�ĺ�б�Ĳ��֣�Ҳ���ǣ�����ֻ�������ǵ��յ㲿�֡�
		findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//��ʾ����
		drawAndShowContours(imgThresh.size(), contours, "imgContours");

		vector<vector<Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			cv::convexHull(contours[i], convexHulls[i]);         //Ѱ��͹��
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

		imgFrame2Copy = imgFrame2.clone();          //������Ĵ����У����Ǹı���֮ǰ��֡2�������õ���һ�����2

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

		imgFrame1 = imgFrame2.clone();           //����1֡�ƶ�����2֡��λ��

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

	if (chCheckForEscKey != 27) {               //����û�û�а���esc(����Ƶ�Ľ���)
		cv::waitKey(1);                         //�ý�����Ϣ��ʾ�ڴ�����
	}

}

