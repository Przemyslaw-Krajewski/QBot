/*
 * PozyskiwaczPulpitu.cpp
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#include "../Analyzers/DesktopHandler.h"

DesktopHandler* DesktopHandler::ptr = nullptr;
Display* DesktopHandler::display = nullptr;
Window DesktopHandler::root = 0;
Window DesktopHandler::window = 0;
int DesktopHandler::revert = 0;

xdo_t* DesktopHandler::xdo = nullptr;

/*
 *
 */
DesktopHandler::DesktopHandler()
{
	display = XOpenDisplay(NULL);
	root = DefaultRootWindow(display);
	XGetInputFocus(display, &window, &revert);
	std::cout << "Window ID:" << window << "\n";

	xdo = xdo_new(NULL);
}

/*
 *
 */
DesktopHandler::~DesktopHandler()
{
	XDestroyWindow(display,root);
}

/*
 *
 */
DesktopHandler* DesktopHandler::getPtr()
{
	if(ptr == nullptr) ptr = new DesktopHandler();
	return ptr;
}

/*
 *
 */
std::pair<int,int> DesktopHandler::getDesktopSize()
{
	XWindowAttributes gwa;
	XGetWindowAttributes(display, root, &gwa);
	return std::pair<int,int>(gwa.width,gwa.height);
}

/*
 *
 */
std::pair<int,int> DesktopHandler::getGameScreenSize()
{
	return std::pair<int,int>(551,446);
}

/*
 *
 */
void DesktopHandler::getDesktop(cv::Mat * mat)
{
	//Potrzebne parametry
	int width = mat->cols;
	int height = mat->rows;
	int channels = mat->channels();
	std::pair<int,int> desktopSize = getDesktopSize();

	//pobranie obrazu
	XImage *dImage = XGetImage(display, root, 0, 0, desktopSize.first, desktopSize.second, AllPlanes, ZPixmap);

	//przepisanie do Matu
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
		 uchar* ptr = mat->ptr(y)+x*channels;
		 *(ptr+0) = dImage->data[((x+(y+90)*desktopSize.first)*4)+0];//blue
		 *(ptr+1) = dImage->data[((x+(y+90)*desktopSize.first)*4)+1];//green
		 *(ptr+2) = dImage->data[((x+(y+90)*desktopSize.first)*4)+2];//red
		}
	}
	XDestroyImage(dImage);
}

/*
 *
 */
void DesktopHandler::pressControllerButton(std::vector<bool> b)
{
	releaseControllerButton(b);
	holdControllerButton(b);
}

/*
 *
 */
void DesktopHandler::holdControllerButton(std::vector<bool> b)
{
	if(b[0])
	{
		const char buttons[] = {'z',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
	if(b[1] && 0)
	{
		const char buttons[] = {'x',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
	if(b[2])
	{
		const char buttons[] = {'L','e','f','t',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
	if(b[3])
	{
		const char buttons[] = {'D','o','w','n',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
	if(b[4])
	{
		const char buttons[] = {'R','i','g','h','t',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
	if(b[5])
	{
		const char buttons[] = {'U','p',0};
		xdo_send_keysequence_window_down(xdo,window,buttons, 20);
	}
}

/*
 *
 */
void DesktopHandler::releaseControllerButton()
{
	const char a[] = {'z','+','x','+','U','p','+','R','i','g','h','t','+','D','o','w','n','+','L','e','f','t',0};

	xdo_send_keysequence_window_up(xdo,window,a, 20);
}

/*
 *
 */
void DesktopHandler::releaseControllerButton(std::vector<bool> b)
{
	if(!b[0])
	{
		const char buttons[] = {'z',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}
	if(!b[1])
	{
		const char buttons[] = {'x',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}
	if(!b[2])
	{
		const char buttons[] = {'L','e','f','t',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}
	if(!b[3])
	{
		const char buttons[] = {'D','o','w','n',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}
	if(!b[4])
	{
		const char buttons[] = {'R','i','g','h','t',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}
	if(!b[5])
	{
		const char buttons[] = {'U','p',0};
		xdo_send_keysequence_window_up(xdo,window,buttons, 20);
	}

}

/*
 *
 */
void DesktopHandler::loadGame()
{
	const char a[] = {'F','7',0};
	xdo_send_keysequence_window_down(xdo,window,a, 20);
	xdo_send_keysequence_window_up(xdo,window,a, 20);
	cv::waitKey(300);
}
