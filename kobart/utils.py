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
import sys
import requests
import hashlib
from zipfile import ZipFile
from transformers import PreTrainedTokenizerFast

tokenizer = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobart/kobart_base_tokenizer_cased_a432df8fec.zip',
    'fname': 'kobart_base_tokenizer_cased_a432df8fec.zip',
    'chksum': 'a432df8fec'
}


def download(url, filename, chksum, cachedir='~/kogpt2/'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path


def get_kobart_tokenizer(cachedir='~/kobart/'):
    """Get KoGPT2 Tokenizer file path after downloading
    """
    global tokenizer
    model_info = tokenizer
    file_path = download(model_info['url'],
                         model_info['fname'],
                         model_info['chksum'],
                         cachedir=cachedir)
    cachedir_full = os.path.expanduser(cachedir)
    if not os.path.exists(os.path.join(cachedir_full, 'emji_tokenizer')):
        zipf = ZipFile(os.path.expanduser(file_path))
        zipf.extractall(path=cachedir_full)
    tok_path = os.path.join(cachedir_full, 'emji_tokenizer/model.json')
    tokenizer_obj = PreTrainedTokenizerFast(tokenizer_file=tok_path,
                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                            pad_token='<pad>', mask_token='<mask>')
    return tokenizer_obj
