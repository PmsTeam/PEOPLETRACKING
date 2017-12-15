#include "PeopleDetectionTracking.h"
extern double accSpeed;
void peopleDetectionTracking(const string videoPath,const string outputPath)
{


	VideoCapture capVideo;

	capVideo.open(videoPath);

	if (!capVideo.isOpened()) 
	{
		cout << "����û�ж�ȡ����Ƶ�ļ���" << endl;
		return;
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2)
	{
		cout << "������Ƶ֡��С��2��" << endl;
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


	Mat imgFrame1;
	Mat imgFrame2;
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	//�����������λ��
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
	int out = 0;//ÿ30֡���һ�����ʱ����ͨ���ĳ���
	ofstream outfile("./cache/out-people.txt");

	//�����ļ������ 
	ofstream oFile;

	//��Ҫ������ļ� 
	oFile.open("./cache/������ϢժҪ.csv", ios::out | ios::trunc);    // �����ͺ����׵����һ����Ҫ��excel �ļ�
	oFile << "��ǰ��Ƶʱ��" << "," << "�ۼ�ͨ����������" << "," << "1s��ͨ����������" << "," << "����ͨ��ƽ������" << endl;
	
	while (capVideo.isOpened() && chCheckForEscKey != 27)
	{
		

		//��ȡ��ǰ֡�ľ�����Ƶ��ʼ��ʱ��λ�ã�ms��
		//frameTime.push_back(capVideo.get(CV_CAP_PROP_POS_MSEC));
		if (out++ == 30)
		{
			peopleDCount = peopleCount - peopleDCount;
			string currentTime = to_string(capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000);
			string subCurrentTime = currentTime.substr(0, currentTime.size() - 5) + "��";
			string avgSpeed;
			if (peopleDCount != 0)
				avgSpeed = to_string(accSpeed / peopleDCount);
			else
				avgSpeed = to_string(0);
			string subAvgSpeed = avgSpeed.substr(0, avgSpeed.size() - 5) + "km/h";

			outfile << "REAL-TIME��";
			outfile << subCurrentTime;
			outfile << "    ";
			outfile << "TOTAL-PEOPLE��";
			outfile << peopleCount;
			outfile << "    ";
			outfile << "REAL-TIME: ";
			outfile << capVideo.get(CV_CAP_PROP_POS_MSEC) / 1000;
			outfile << "    ";
			outfile << "REAL-PEOPLE: ";
			outfile << peopleDCount << endl;
			//��ϵ�ǰʱ�䣬д������������
			oFile << subCurrentTime << "," << peopleCount << "," << peopleDCount << "," << subAvgSpeed << endl;
			peopleDCount = peopleCount;
			accSpeed = 0;
			out = 0;
		}

		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();
		Mat imgDifference;
		Mat imgThresh;

		//������������ģ�����±���
		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);
		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		
		//��ö�ֵ��ͼ��
		threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
		//imshow("imgThresh", imgThresh);


		//��ȡ���õĽṹԪ�ص���״
		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement15x15 = getStructuringElement(MORPH_RECT, Size(15, 15));

		//�������ų�С�ͺڶ�
		for (unsigned int i = 0; i < 2; i++)
		{
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}

		Mat imgThreshCopy = imgThresh.clone();


		//Ѱ�Ҳ�����������ÿ�������洢Ϊһ��������
		vector<vector<Point> > contours;
		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//drawAndShowContours(imgThresh.size(), contours, "imgContours");

		//Ѱ�Ҳ�����͹��
		vector<vector<Point> > convexHulls(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++)
			convexHull(contours[i], convexHulls[i]);

		vector<Blob> currentFrameBlobs; //��ǰ֡�е��ſ�

		//drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
		//�ж��ſ��Ƿ�Ϊ����
		for (auto &convexHull : convexHulls)
		{
			Blob possibleBlob(convexHull);
			possibleBlob.judgePeople(possibleBlob);
			if (possibleBlob.blnIsPeople)
				currentFrameBlobs.push_back(possibleBlob);
		}
		//drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		//����ǰ�����ſ�����ܳ����ſ鼯��
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


		/*��ȡ��һ֡ͼ��*/
		currentFrameBlobs.clear();
		imgFrame1 = imgFrame2.clone();
		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT))
		{
			capVideo.read(imgFrame2);
		}
		else
		{
			cout << "��Ƶ����" << endl;
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