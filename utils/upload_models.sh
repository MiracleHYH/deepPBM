#!/bin/bash

zipFileName=models.zip
uploadFolder="http://sdly.blockelite.cn:15021/web/client/pubshares/5yoVa5SKmBQhZjKvQ7KfRo/"

zip -u zipFileName ../models/*.pth
curl --data-binary @zipFileName -H "Content-Type: application/octet-stream" -H "X-SFTPGO-MTIME: 1638882991234" ${uploadFolder}${zipFileName}
rm zipFileName