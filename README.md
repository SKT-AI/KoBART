
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [🤣 KoBART](#kobart)
  - [How to install](#how-to-install)
  - [Data](#data)
  - [Tokenizer](#tokenizer)
  - [Model](#model)
    - [Performances](#performances)
  - [Contacts](#contacts)
  - [License](#license)

<!-- /code_chunk_output -->


# 🤣 KoBART

[**BART**](https://arxiv.org/pdf/1910.13461.pdf)(**B**idirectional and **A**uto-**R**egressive **T**ransformers)는 입력 텍스트 일부에 노이즈를 추가하여 이를 다시 원문으로 복구하는 `autoencoder`의 형태로 학습이 됩니다. 한국어 BART(이하 **KoBART**) 논문에서 언급한 `Text Infilling` 노이즈 함수를 사용하여 **40GB** 이상의 한국어 텍스트에 대해서 학습한 한국어 `encoder-docoder` 언어 모델입니다. 이를 통해 도출된 `KoBART-base`을 배포합니다.


![](imgs/bart.png)

## How to install

```
git clone https://github.com/SKT-AI/KoBART.git
cd KoBART
pip install -r requirements.txt
pip install .
```

## Data

| Data  | # of Sentences |
|-------|---------------:|
| Korean Wiki |     5M   |  
| Other corpus |  0.27B    | 

한국어 wiki 데이터 이외 뉴스, books, 모두의 코퍼스(대화, 뉴스, ...), 청와대 국민청원 등의 다양한 데이터가 사용되었습니다.

## Tokenizer

[`tokenizers`](https://github.com/huggingface/tokenizers) 패키지의 `Character BPE tokenizer`로 학습되었습니다. 

`vocab` 사이즈는 30,000 이며 대화에 자주 쓰이는 아래와 같은 이모티콘, 이모지 등을 추가하여 해당 토큰의 인식 능력을 올렸습니다. 
> 😀, 😁, 😆, 😅, 🤣, ,..., `:-)`, `:)`, `-)`, `(-:`...

또한 `<unused0>` ~ `<unused99>`등의 미사용 토큰을 정의해 필요한 `subtasks`에 따라 자유롭게 정의해 사용할 수 있게 했습니다.


```python
>>> from kobart import get_kobart_tokenizer
>>> kobart_tokenizer = get_kobart_tokenizer()
>>> kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.")
['▁안녕하', '세요.', '▁한국어', '▁B', 'A', 'R', 'T', '▁입', '니다.']
```

## Model

| Model         |  Type   | n_layers  | n_heads | ffn_dim | hidden_dims | 
|---------------|:-------:|--------:|--------:|--------:|--------------:|
| `KoBART-base` | Encoder |   6     | 16      | 3072    | 768           |
|               | Decoder |   6     | 16      | 3072    | 768           |


```python
>>> from transformers import BartModel
>>> from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
>>> kobart_tokenizer = get_kobart_tokenizer()
>>> model = BartModel.from_pretrained(get_pytorch_kobart_model())
>>> inputs = kobart_tokenizer(['안녕하세요.'], return_tensors='pt')
>>> model(inputs['input_ids'])
Seq2SeqModelOutput(last_hidden_state=tensor([[[-1.5372, -2.5599,  0.8382,  ..., -2.6832,  2.5374,  1.7316],
         [-1.6075, -3.0245,  1.3806,  ..., -3.4531,  1.8102,  2.0583]]],
       grad_fn=<TransposeBackward0>), past_key_values=None, decoder_hidden_states=None, decoder_attentions=None, cross_attentions=None, encoder_last_hidden_state=tensor([[[ 0.5163, -0.3525,  0.5279,  ...,  0.1081,  0.5969,  0.1189],
         [ 0.4078, -0.3281,  0.6627,  ...,  0.0751,  0.6414,  0.3749]]],
       grad_fn=<TransposeBackward0>), encoder_hidden_states=None, encoder_attentions=None) 
```

### Performances

|   |  NSMC(acc)  | KorSTS(spearman) | Question Pair(acc) | 
|---|---|---|---|
| **KoBART**  | 90.07  | 81.31  | 93.80  |


## Contacts

`KoBART` 관련 이슈는 [이곳](https://github.com/SKT-AI/KoBART/issues)에 올려주세요.

## License

`KoBART`는 `modified MIT` 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 `LICENSE` 파일에서 확인하실 수 있습니다.
