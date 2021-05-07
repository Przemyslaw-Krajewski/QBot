/*
 * RawImageAnalyzer.h
 *
 *  Created on: 5 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_RAWIMAGEANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_RAWIMAGEANALYZER_H_

#include "ImageAnalyzer.h"

class RawImageAnalyzer : public ImageAnalyzer {

public:

	RawImageAnalyzer(Game t_game);
	virtual ~RawImageAnalyzer();

	void processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result);
	virtual std::vector<int> createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
												  ControllerInput& controllerInput, Point& position, Point& velocity) override;
	virtual State reduceSceneState(const State& t_state, double action) override;
protected:

	void calculateSituationSMB(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult);
	void calculateSituationBT(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult);

private:

	void reduceColors(int mask, cv::Mat* colorImage);
	void reduceColorsAndBrightness(int reduceLevel, cv::Mat* colorImage);
	void getMostFrequentInBlock(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage);
	void getLeastFrequentInImage(int blockSize, cv::Mat& srcIimage, cv::Mat& dstImage);
	cv::Mat getFirst(int blockSize, cv::Mat* image);

	cv::Point imageSize;

	std::list<cv::Mat> oldImages;
	std::list<cv::Mat> oldImages2;

	//SMB
	cv::Mat deadImage;
	cv::Mat winImage;

	//BT
	cv::Mat emptyHealth;
	cv::Mat hair;

	int holdButtonCounter;
};

#endif /* SRC_ANALYZERS_IMAGEANALYZER_RAWIMAGEANALYZER_H_ */
