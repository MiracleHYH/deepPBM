#!/bin/bash

Model=models.zip
Log=logs.zip
Folder="http://sdly.blockelite.cn:15021/web/client/pubshares/5yoVa5SKmBQhZjKvQ7KfRo/"

zip -u -r ${Model} ../models/*.pth
zip -u -r ${Log} ../logs/*
curl --data-binary @${Model} -H "Content-Type: application/octet-stream" -H "X-SFTPGO-MTIME: 1638882991234" ${Folder}${Model}
curl --data-binary @${Log} -H "Content-Type: application/octet-stream" -H "X-SFTPGO-MTIME: 1638882991234" ${Folder}${Log}
rm ${Model}
rm ${Log}