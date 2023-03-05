#!/bin/bash

fileName=BMC2012_real_video.zip
dataURL="http://sdly.blockelite.cn:15021/web/client/pubshares/ZMVHj9ciyt2KsHdJ8MNyaW?compress=false"

wget "$dataURL" -O $fileName
unzip -u $fileName -d ../data
rm $fileName