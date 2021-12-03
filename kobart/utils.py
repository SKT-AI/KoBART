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
import shutil
import sys
from zipfile import ZipFile

import requests
from transformers import PreTrainedTokenizerFast


tokenizer = {
    "url": "https://kobert.blob.core.windows.net/models/kobart/kobart_base_tokenizer_cased_cf74400bce.zip",
    "fname": "kobart_base_tokenizer_cased_cf74400bce.zip",
    "chksum": "cf74400bce",
}


def download(url, filename, chksum, cachedir=".cached"):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10] == chksum:
            print("using cached model")
            return file_path, True
    with open(file_path, "wb") as f:
        response = requests.get(url, stream=True)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                chunk_size=max(int(total / 1000), 1024 * 1024)
            ):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write("\r[{}{}{}]".format(file_path, "█" * done, "." * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write("\n")
    assert (
        chksum == hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10]
    ), "corrupted file!"
    return file_path, False


def get_kobart_tokenizer(cachedir=".cache"):
    """Get KoGPT2 Tokenizer file path after downloading"""
    global tokenizer
    model_info = tokenizer
    file_path, is_cached = download(
        model_info["url"], model_info["fname"], model_info["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    if (
        not os.path.exists(os.path.join(cachedir_full, "emji_tokenizer"))
        or not is_cached
    ):
        if not is_cached:
            shutil.rmtree(
                os.path.join(cachedir_full, "emji_tokenizer"), ignore_errors=True
            )
        zipf = ZipFile(os.path.expanduser(file_path))
        zipf.extractall(path=cachedir_full)
    tok_path = os.path.join(cachedir_full, "emji_tokenizer/model.json")
    tokenizer_obj = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    return tokenizer_obj
