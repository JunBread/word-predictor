# word-predictor

News headline word predictor using LSTM. This is a sub-project of [skku-coop-project](https://github.com/JunBread/skku-coop-project).

## File Description

`predictor.py`: Main python file. To predict headline which starts with a given word, type like this:

```bash
python predictor.py 청와대
```

`model.h5`: Pretrained model using sample `news-title.txt` file. The predictor will use pretrained model if this file exists. If you want to train your own model, remove it.

`news-title.txt`: News headline dataset. Crawled from [BigKinds](https://www.bigkinds.or.kr) news service. Each line must end with a special tag, `<E>`.
