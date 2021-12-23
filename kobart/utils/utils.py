# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2020 SK telecom

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import hashlib
import os

from kobart.utils.aws_s3_downloader import AwsS3Downloader


def download(url, chksum=None, cachedir=".cache"):
    cachedir_full = os.path.join(os.getcwd(), cachedir)
    os.makedirs(cachedir_full, exist_ok=True)
    filename = os.path.basename(url)
    file_path = os.path.join(cachedir_full, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10] == chksum:
            print(f"using cached model. {file_path}")
            return file_path, True

    s3 = AwsS3Downloader()
    file_path = s3.download(url, cachedir_full)
    if chksum:
        assert (
            chksum == hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10]
        ), "corrupted file!"
    return file_path, False
