#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pyplot

colors = ['k','r','g','b','c','m','y']

def plotFigure(fileName):
    score=np.loadtxt(fileName)
    average=score.copy()
   
    av=score[0]
    for i,v in enumerate(score):
        av=av+(v-av)*0.1
        average[i]=av

    pyplot.title('Updated figure')
    pyplot.plot(range(len(score)),score,'b')
    pyplot.plot(range(len(average)),average,'r')

def plotFigures(fileNames):
    for f,fileName in enumerate(fileNames):
        score=np.loadtxt(fileName)
        average=score.copy()
   
        av=score[0]
        for i,v in enumerate(score):
            av=av+(v-av)*0.1
        average[i]=av

        pyplot.title('All figures')
        pyplot.plot(range(len(score)),score,colors[f],label=fileName)
    pyplot.legend()
    pyplot.show()


if len(sys.argv)==1:
    pyplot.ion()
    pyplot.show()
    while True:
        plotFigure("../score.log")
        pyplot.pause(5)
else:
    plotFigures(sys.argv[1:])
