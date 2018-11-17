/*
 * PozyskiwaczPulpitu.h
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#ifndef SRC_IMAGEANALYZER_OBSLUGAPULPITU_H_
#define SRC_IMAGEANALYZER_OBSLUGAPULPITU_H_

#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

extern "C" {
#include <xdo.h>
}


class DesktopHandler {

private:
	DesktopHandler();
	~DesktopHandler();

public:
	static DesktopHandler* getPtr();

	static void getDesktop(cv::Mat * mat);
	static std::pair<int,int> getDesktopSize();
	static std::pair<int,int> getGameScreenSize();

	static void pressControllerButton(std::vector<bool> b);
	static void holdControllerButton(std::vector<bool> b);
	static void releaseControllerButton();
	static void releaseControllerButton(std::vector<bool> b);
	static void loadGame();

private:
	static DesktopHandler* ptr;
	static Display *display;
	static Window root;
	static Window window;
	static int revert;

	static xdo_t *xdo;
};

#endif /* SRC_IMAGEANALYZER_OBSLUGAPULPITU_H_ */
