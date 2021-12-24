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
from transformers import PreTrainedTokenizerFast

from kobart import download


def get_pytorch_kobart_model(ctx="cpu", cachedir=".cache"):
    pytorch_kobart = {
        "url": "s3://skt-lsl-nlp-model/KoBART/models/kobart_base_cased_ff4bda5738.zip",
        "chksum": "ff4bda5738",
    }
    model_zip, is_cached = download(
        pytorch_kobart["url"], pytorch_kobart["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.join(os.getcwd(), cachedir)
    model_path = os.path.join(cachedir_full, "kobart_from_pretrained")
    if not os.path.exists(model_path) or not is_cached:
        if not is_cached:
            shutil.rmtree(model_path, ignore_errors=True)
        zipf = ZipFile(os.path.expanduser(model_zip))
        zipf.extractall(path=cachedir_full)
    return model_path


def get_kobart_tokenizer(cachedir=".cache"):
    """Get KoGPT2 Tokenizer file path after downloading"""
    tokenizer = {
        "url": "s3://skt-lsl-nlp-model/KoBART/tokenizers/kobart_base_tokenizer_cased_cf74400bce.zip",
        "chksum": "cf74400bce",
    }
    file_path, is_cached = download(
        tokenizer["url"], tokenizer["chksum"], cachedir=cachedir
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


if __name__ == "__main__":
    # pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
    from transformers import BartModel
    from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

    kobart_tokenizer = get_kobart_tokenizer()
    print(kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.🤣:)l^o"))

    model = BartModel.from_pretrained(get_pytorch_kobart_model())
    inputs = kobart_tokenizer(["안녕하세요."], return_tensors="pt")
    print(model(inputs["input_ids"]))
