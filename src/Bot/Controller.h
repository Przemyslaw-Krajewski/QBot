/*
 * Controller.h
 *
 *  Created on: 19 wrz 2021
 *      Author: przemo
 */

#ifndef SRC_BOT_CONTROLLER_H_
#define SRC_BOT_CONTROLLER_H_

const int numberOfControllerInputs = 6;

using ControllerInput = std::vector<bool>;

class Controller
{
public:

	Controller() : Controller(0)
	{

	}

	Controller(int t_action)
	{
		action = t_action;

		switch(t_action)
		{
		case 0:
			code = RIGHT;
			break;
		case 1:
			code = RIGHT+A;
			break;
		default:
			code = NOTHING;
		}

		input = ControllerInput(6,false);
		input[0] = (code&A)!=0;
		input[1] = (code&B)!=0;
		input[2] = (code&UP)!=0;
		input[3] = (code&DOWN)!=0;
		input[4] = (code&LEFT)!=0;
		input[5] = (code&RIGHT)!=0;
	}

	int getAction() {return action;}
	int getCode() {return code;}
	ControllerInput& getInput() {return input;}

private:
	ControllerInput input;
	int code;
	int action;

public:
	static constexpr int NOTHING=0;
	static constexpr int A=1;
	static constexpr int B=2;
	static constexpr int UP=16;
	static constexpr int DOWN=32;
	static constexpr int LEFT=64;
	static constexpr int RIGHT=128;
};

#endif /* SRC_BOT_CONTROLLER_H_ */
