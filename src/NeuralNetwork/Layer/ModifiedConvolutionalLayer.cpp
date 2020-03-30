//
// Created by przemo on 02.01.2020.
//

#include "ModifiedConvolutionalLayer.h"

/*
 *
 */
ModifiedConvolutionalLayer::ModifiedConvolutionalLayer(double t_learnRate, MatrixSize t_filterSize, int t_numberOfLayers,
													   TensorSize t_inputSize, std::vector<Neuron*> t_prevLayerReference) :
													   	   filterSize(t_filterSize.x, t_filterSize.y, t_inputSize.z),
														   outputSize(t_inputSize.x, t_inputSize.y, t_numberOfLayers)
{
    emptyValue = InputNeuron(0.0);
    learnRate = t_learnRate;

    t_filterSize.x = (t_filterSize.x-1) / 2;
    t_filterSize.y = (t_filterSize.y-1) / 2;
    for(int z=0; z<t_numberOfLayers; z++)
    {
        for(int y=0; y<t_inputSize.y; y+=1)
        {
            for (int x=0; x<t_inputSize.x; x+=1)
            {
                std::vector<Neuron*> neuronsReference;
                for(int fz=0; fz<t_inputSize.z; fz++)
                {
                    for (int fy=-t_filterSize.y; fy < t_filterSize.y; fy++)
                    {
                        for (int fx=-t_filterSize.x; fx < t_filterSize.x; fx++)
                        {
                            if(x-fx<0 || x-fx>=t_inputSize.x || y-fy<0 || y-fy>=t_inputSize.y)
                                neuronsReference.push_back(&emptyValue);
                            else
                                neuronsReference.push_back(t_prevLayerReference[getIndex(x - fx, y - fy, fz, t_inputSize.x,t_inputSize.y)]);
                        }
                    }
                }

                neurons.emplace_back(neuronsReference, &learnRate,
                                     [](double x) -> double { return x > 0 ? x : 0.01*x; },
                                     [](double x) -> double { return x > 0 ? 1 : 0.01; });

//                [](double x) -> double { return 1 / ( 1 + exp(-0.53* x) ); },
//                                [](double x) -> double { double e = exp(-0.53*x);
//                                      double m = 1 + e;
//                                      return -(0.53*e/(m*m));});
            }
        }
    }
}

/*
 *
 */
std::vector<double> ModifiedConvolutionalLayer::getOutput()
{
    std::vector<double> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(it->getOutput());
    }
    return result;
}

void ModifiedConvolutionalLayer::determineOutput()
{
    int i;
    #pragma omp parallel for shared(neurons) private(i) default(none)
    for(i=0; i<neurons.size(); i++)
    {
        neurons[i].determineOutput();
    }
}


/*
 *
 */
void ModifiedConvolutionalLayer::setDelta(std::vector<double> t_z)
{
    assert(t_z.size() == neurons.size() && "learning values size not match");
    int i=0;
    for( auto it = neurons.begin(); it != neurons.end(); it++,i++)
    {
        it->setDelta(t_z[i]-it->getOutput());
    }
}

/*
 *
 */
void ModifiedConvolutionalLayer::learnBackPropagation()
{
//	int64 timeBefore = cv::getTickCount();
    int i;
	#pragma omp parallel for shared(neurons) private(i) default(none)
    for(i=0; i<neurons.size(); i++)
    {
        neurons[i].learnDeltaRule();
    }
//	int64 afterBefore = cv::getTickCount();
//	std::cout << "Conv: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
}

/*
 *
 */
std::vector<Neuron *> ModifiedConvolutionalLayer::getNeuronPtr()
{
    std::vector<Neuron *> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(&(*it));
    }

    return result;
}

/*
 *
 */
void ModifiedConvolutionalLayer::drawLayer()
{
	int blockSize = 4;
	std::vector<double> result = getOutput();
	for(int z = 0; z< outputSize.z ;z++)
	{
		//View
		cv::Mat viewImage = cv::Mat((outputSize.y)*blockSize, (outputSize.x)*blockSize, CV_8UC3);
		for(int x=0; x<outputSize.x; x++)
		{
			for(int y=0; y<outputSize.y; y++)
			{
				cv::Scalar color;
				int value = result[getIndex(x,y,z,outputSize.x,outputSize.y)];
				for(int yy=0; yy<blockSize; yy++)
				{
					for(int xx=0; xx<blockSize; xx++)
					{
						uchar* ptr = viewImage.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
						if(value > 0) {ptr[0] = 0;ptr[1] = value;ptr[2] = 0;}
						else {ptr[0] = 0;ptr[1] = 0;ptr[2] = abs(value);}
					}
				}
			}
		}
		//Print
		std::string a = std::to_string(z);
		imshow(a, viewImage);
		cv::waitKey(10);
	}
}
