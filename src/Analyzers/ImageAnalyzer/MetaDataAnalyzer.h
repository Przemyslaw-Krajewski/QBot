/*
 * MetaDataAnalyzer.h
 *
 *  Created on: 5 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_METADATAANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_METADATAANALYZER_H_

#include "ImageAnalyzer.h"

class MetaDataAnalyzer : public ImageAnalyzer {
public:
	MetaDataAnalyzer(Game t_game);
	virtual ~MetaDataAnalyzer();

	virtual void processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result) override;
	virtual State createSceneState(std::vector<cv::Mat> &t_images, ControllerInput& t_controllerInput, Point& t_position, Point& t_velocity) override;
	virtual void correctScenarioHistory(std::list<SARS> &t_history, bool t_killedByEnemy) override;
private:

	void processSMBImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result);

	cv::Point findPlayer(cv::Mat &image);
	Histogram determineHistogram(cv::Mat &image);
	cv::Mat copyMat(cv::Mat src, cv::Point offset, cv::Point size);
	bool compareMat(cv::Mat &mat1, cv::Mat &mat2);
	void markObjectInImage(cv::Mat& resultImage, cv::Point blockSize, cv::Point point, cv::Point translation, cv::Point correction, int objectType);


	//SMB
	cv::Mat enemyImage1,enemyImage1v2;
	cv::Mat enemyImage2,enemyImage2v2;
	cv::Mat floorImage1,floorImage1v2;
	cv::Mat wallimage1,wallImage1v2;
	cv::Mat blockImage1,blockImage1v2;
	cv::Mat blockImage2,blockImage2v2;
	cv::Mat blockImage3,blockImage3v2;
	cv::Mat pipeImage;
	cv::Mat deadImage;
	cv::Mat winImage;
	cv::Mat mushroomImage;
	cv::Mat cloudImage;

	cv::Point playerSize;
	Histogram playerHistogram;

	int holdButtonCounter;

};

#endif /* SRC_ANALYZERS_IMAGEANALYZER_METADATAANALYZER_H_ */
