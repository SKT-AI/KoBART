# coding=utf-8
# Copyright 2020 SKT T3K Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from zipfile import ZipFile

import torch
from transformers import BartModel
from .utils import download as _download

pytorch_kobart = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobart/kobart_base_cased_12697364e2.zip',
    'fname': 'kobart_base_cased_12697364e2.zip',
    'chksum': '12697364e2'
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

