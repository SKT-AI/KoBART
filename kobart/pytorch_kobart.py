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

import os
from zipfile import ZipFile

from .utils import download as _download

pytorch_kobart = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobart/kobart_base_cased_e2b7dcdef4.zip',
    'fname': 'kobart_base_cased_e2b7dcdef4.zip',
    'chksum': 'e2b7dcdef4'
}


def get_pytorch_kobart_model(ctx='cpu', cachedir='~/kobart/'):
    # download model
    global pytorch_kobart
    model_info = pytorch_kobart
    model_zip = _download(model_info['url'],
                          model_info['fname'],
                          model_info['chksum'],
                          cachedir=cachedir)
    cachedir_full = os.path.expanduser(cachedir)
    if not os.path.exists(os.path.join(cachedir_full, 'kobart_emji_from_pretrained')):
        zipf = ZipFile(os.path.expanduser(model_zip))
        zipf.extractall(path=cachedir_full)
    model_path = os.path.join(cachedir_full, 'kobart_emji_from_pretrained')
    return model_path
