Seq2Seq + Attension Chatbot
===========================

sequense-to-sequenseモデルを活用した自動応答アプリケーション

## コーパスのダウンロード
[OpenSubtitles](https://www.opensubtitles.org/)から、映画の字幕データをダウンロードし、会話の配列を作成します

```py
import datasets
data = datasets.open_subtitles(download=False)
> [
>   ['sentense1', 'sentense2', 'sentense3', ...],
>   ['sentense1', 'sentense2', 'sentense3', ...],
>   ...
> ]
```

## 単語ベクトルへ変換
NLTKを用いて、文章を単語列に変換します。その後、単語列から辞書を作成し、単語列を辞書のインデックスに変換します。

```py
import preprocessing
processor = preprocessing.PreProcessing()
token_vectors, dictionary = processor.generate_token_vectors(data, max_len=MAX_LENGTH)
token_vectors
> [
>   [
>     [322, 121, 12312, 222, ..., 5, 0, 0, 0], # sentense1
>     [322, 121, 12312, 222, ..., 5, 0, 0, 0], # sentense2
>     ...
>   ],
>   [
>     [322, 121, 12312, 222, ..., 5, 0, 0, 0], # sentense1
>     [322, 121, 12312, 222, ..., 5, 0, 0, 0], # sentense2
>     ...
>   ],
>   ...
> ]
dictionary['word2idx']
> {'': 0, 'hello': 1, 'world': 2, ...}
```

## 学習

```py
dialog = models.Dialog(dictionary, MAX_LENGTH, hidden_dim=n_hidden, depth=depth)
model = dialog.create_model()
for vectors in token_vectors:
    x_train, x_test, y_test, y_train = dialog.get_training_batch(vectors)
    dialog.train(x_train, y_train,
                 batch_size=batch_size, nb_epoch=nb_epoch,
                 validation_data=(x_test, y_test),
                 save_model=True)
```

## 応答テスト

```py
message = dialog.reply("hello")
message
> "hi, what's wrong?"
message = dialog.reply("tell me how to use")
message
> "later, please"
```
