#!/bin/bash

for (( i=1; i<=$1; ++i));
do
	sleep 60
	echo $i
done
touch ../build/quit.param
