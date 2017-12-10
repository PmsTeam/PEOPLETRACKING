//ͷ�ļ��Լ������ռ�//////////////////////////////////////////////////////////////
#include<string.h>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<conio.h>          // �����ʹ��Windows��������Ҫ���Ļ�ɾ������

#include "Blob.h"

#define SHOW_STEPS           // ȡ��ע�ͻ�ע�ʹ�������ʾ����

// ȫ�ֱ���������ɫ�Ķ���///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// ����ԭ�� ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
bool Aspectratio1(Blob frameL);
bool Aspectratio2(Blob frameL);
bool checkIfBlobsCrossedTheLine2(std::vector<Blob> &blobs, int &intHorizontalLinePosition2, int &peopleCount);
void drawPeopleCountOnImage(int &peopleCount, cv::Mat &imgFrame2Copy);
//main()���������￪ʼ/////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
	//cv::VideoCapture cap;
	cv::VideoCapture capVideo;

	cv::Mat imgFrame1;
	cv::Mat imgFrame2;

	std::vector<Blob> blobs;

	cv::Point crossingLine[2];
	cv::Point crossingLine2[2];

	int carCount = 0;
	int peopleCount = 0;

	//capVideo = cap;
	capVideo.open("2.avi");

	// �������Ƶʧ��
	if (!capVideo.isOpened()) {
		//����������ʾ
		std::cout << "error reading video file" << std::endl << std::endl;
		// �����ʹ��Windows��������Ҫ���Ļ�ɾ������
		_getch();
		return(0);
	}

	//capVideo.get(CV_CAP_PROP_FRAME_COUNT) Ϊ��Ƶ�ļ��е�֡��
	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		//�����Ƶ��֡��С��2������������Ϣ
		std::cout << "error: video file must have at least two frames";
		// �����ʹ��Windows��������Ҫ���Ļ�ɾ������
		_getch();
		return(0);
	}

	//ͬcaptuer >> imgFrame ����Ƶ֡����
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	//���ߺ�������������ͳ����/////////////////////////////////////////
	int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.529);         //��
	crossingLine[0].x = imgFrame1.cols * 0.085;
	crossingLine[0].y = imgFrame1.rows * 0.529;
	crossingLine[1].x = imgFrame1.cols * 0.386;
	crossingLine[1].y = imgFrame1.rows * 0.529;

	int intHorizontalLinePosition2 = (int)std::round((double)imgFrame1.rows * 0.296);    //��
	crossingLine2[1].x = imgFrame1.cols * 0.320;
	crossingLine2[1].y = imgFrame1.rows * 0.296;
	crossingLine2[0].x = imgFrame1.cols;
	crossingLine2[0].y = imgFrame1.rows * 0.296;
	///////////////////////////////////////////////////////////////////

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;

	while (capVideo.isOpened() && chCheckForEscKey != 27) {     //capVideo.isOpende()�ж���Ƶ��ȡ��������ͷ�����Ƿ�ɹ����ɹ��򷵻�true��

		std::vector<Blob> currentFrameBlobs;

		//����ԭʼ���ݵĸ���
		cv::Mat imgFrame1Copy = imgFrame1.clone();
		cv::Mat imgFrame2Copy = imgFrame2.clone();

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		//��ɫת����CV_BGR2GRAY ����ɫ��RGB��grayװ��
		cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		//��˹�˲�������cv:Size()Ϊ��˹�ں˴�С��0λ��X,y����ı�׼����
		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		//����ǰ����ͼ��ROI��ֵ�Ĳ�ֵ����ֵ��������ͼ��
		cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		//��ֵ���޷ָ�cv::threshold����������ֱ�Ϊ ����������������ֵ�����õ����ֵ����ֵ����
		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		// cv::imshow("imgThresh", imgThresh);

		//OpenCV�ṩ��һ������getStructuringElement�����Ի�ȡ���õĽṹԪ�ص���״�����Σ��������Σ�MORPH_RECT����Բ������Բ�Σ�MORPH_ELLIPSE��ʮ����MORPH_CROSS��
		cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

		//���ͺ���.
		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		//��ʴ����
		cv::erode(imgThresh, imgThresh, structuringElement5x5);

		//��¡ͼ��
		cv::Mat imgThreshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point> > contours;

		//����cvFindContours�Ӷ�ֵͼ���м��������������ؼ�⵽�������ĸ�����
		//https://baike.baidu.com/item/cvFindContours/9560509?fr=aladdin
		//CV_RETR_EXTERNAL��ֻ�����������������CV_CHAIN_APPROX_SIMPLE��ѹ��ˮƽ�ġ���ֱ�ĺ�б�Ĳ��֣�Ҳ���ǣ�����ֻ�������ǵ��յ㲿�֡�
		cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//��ʾ����
		//drawAndShowContours(imgThresh.size(), contours, "imgContours");

		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			cv::convexHull(contours[i], convexHulls[i]);         //Ѱ��͹��
		}

		// drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);

			if (Aspectratio2(possibleBlob) || Aspectratio1(possibleBlob)) {
				currentFrameBlobs.push_back(possibleBlob);
			}
		}

		// drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true) {
			for (auto &currentFrameBlob : currentFrameBlobs) {
				blobs.push_back(currentFrameBlob);
			}
		}
		else {
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		}

		// drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

		imgFrame2Copy = imgFrame2.clone();          //������Ĵ����У����Ǹı���֮ǰ��֡2�������õ���һ�����2

		drawBlobInfoOnImage(blobs, imgFrame2Copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine2(blobs, intHorizontalLinePosition2, peopleCount);
		if (blnAtLeastOneBlobCrossedTheLine == true) {
			cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_BLACK, 2);
		}
		else {
			cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		}

		blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);
		if (blnAtLeastOneBlobCrossedTheLine == true) {
			cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_GREEN, 2);
		}
		else {
			cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_RED, 2);
		}

		drawCarCountOnImage(carCount, imgFrame2Copy);
		drawPeopleCountOnImage(peopleCount, imgFrame2Copy);
		cv::imshow("imgFrame2Copy", imgFrame2Copy);

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
		cv::waitKey(0);                         //�ý�����Ϣ��ʾ�ڴ�����
	}

	return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

	for (auto &existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();
	}

	for (auto &currentFrameBlob : currentFrameBlobs) {

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			if (existingBlobs[i].blnStillBeingTracked == true) {

				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else {
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	for (auto &existingBlob : existingBlobs) {

		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}

		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
	existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	std::vector<std::vector<cv::Point> > contours;

	for (auto &blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
	bool blnAtLeastOneBlobCrossedTheLine = false;

	for (auto blob : blobs) {

		if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
				carCount++;
				blnAtLeastOneBlobCrossedTheLine = true;
			}
		}

	}

	return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine2(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &peopleCount) {
	bool blnAtLeastOneBlobCrossedTheLine = false;

	for (auto blob : blobs) {

		if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
				peopleCount++;
				blnAtLeastOneBlobCrossedTheLine = true;
			}
		}

	}

	return blnAtLeastOneBlobCrossedTheLine;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {
			cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

			int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
			int intFontThickness = (int)std::round(dblFontScale * 1.0);

			cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;

	ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

	cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_WHITE, intFontThickness);

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawPeopleCountOnImage(int &peopleCount, cv::Mat &imgFrame2Copy) {

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize = cv::getTextSize(std::to_string(peopleCount), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;

	ptTextBottomLeftPosition.x = (int)((double)textSize.width * 1.25);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

	cv::putText(imgFrame2Copy, std::to_string(peopleCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_WHITE, intFontThickness);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool Aspectratio1(Blob possibleBlob)
{
	if (possibleBlob.currentBoundingRect.area() > 2100 &&
		//�����
		possibleBlob.dblCurrentAspectRatio > 1.25 && possibleBlob.dblCurrentAspectRatio < 2.5 &&
		//�߽���εĿ���
		possibleBlob.currentBoundingRect.width > 60 &&
		possibleBlob.currentBoundingRect.height > 30 &&
		//�Խ��߳���
		possibleBlob.dblCurrentDiagonalSize > 60 &&
		//contourArea���ڼ������
		(cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50)
		return true;
	else
		return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
bool Aspectratio2(Blob possibleBlob)
{
	if (possibleBlob.currentBoundingRect.area() > 800 &&
		//�����
		possibleBlob.dblCurrentAspectRatio > 0.35 && possibleBlob.dblCurrentAspectRatio < 0.80 &&
		//�߽���εĿ���
		possibleBlob.currentBoundingRect.width > 20 &&
		possibleBlob.currentBoundingRect.height > 35 &&
		//�Խ��߳���
		possibleBlob.dblCurrentDiagonalSize > 35 &&
		//contourArea���ڼ������
		(cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50)
		return true;
	else
		return false;
}

