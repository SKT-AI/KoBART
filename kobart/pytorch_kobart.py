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
import shutil
from zipfile import ZipFile

from kobart.utils import download as _download

pytorch_kobart = {
    "url": "https://kobert.blob.core.windows.net/models/kobart/kobart_base_cased_ff4bda5738.zip",
    "fname": "kobart_base_cased_ff4bda5738.zip",
    "chksum": "ff4bda5738",
}


def get_pytorch_kobart_model(ctx="cpu", cachedir=".cache"):
    # download model
    global pytorch_kobart
    model_info = pytorch_kobart
    model_zip, is_cached = _download(
        model_info["url"], model_info["fname"], model_info["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    model_path = os.path.join(cachedir_full, "kobart_from_pretrained")
    if not os.path.exists(model_path) or not is_cached:
        if not is_cached:
            shutil.rmtree(model_path, ignore_errors=True)
        zipf = ZipFile(os.path.expanduser(model_zip))
        zipf.extractall(path=cachedir_full)
    return model_path


if __name__ == "__main__":
    # pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
    from transformers import BartModel
    from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

    kobart_tokenizer = get_kobart_tokenizer()
    print(kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.🤣:)l^o"))

    model = BartModel.from_pretrained(get_pytorch_kobart_model())
    inputs = kobart_tokenizer(["안녕하세요."], return_tensors="pt")
    print(model(inputs["input_ids"]))
