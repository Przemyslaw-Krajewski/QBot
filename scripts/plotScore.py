#!/usr/bin/env python

import sys
import time
import numpy as np
import matplotlib.pyplot as pyplot

def plotFigure(fileName, figure,block=True):
    score=np.loadtxt(fileName)
    average=score.copy()
    
    for i in range(len(average)-1,9,-1):
        average[i]=sum(average[i-10:i])/10

    pyplot.figure(figure)
    pyplot.title(fileName)
    pyplot.plot(range(len(score)),score,'b',range(len(average)),average,'r')
    pyplot.show(block=block)

if len(sys.argv)==1:
    plotFigure("../build/score.log",1)
else:
    for i in range(1,len(sys.argv)-1):
        print(sys.argv[i])
        plotFigure(sys.argv[i],i,False)
    plotFigure(sys.argv[-1],len(sys.argv))

