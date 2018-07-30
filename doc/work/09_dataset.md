# （参考）Kerasに付属する学習データセット

Kerasには画像認識に利用できる以下のデータセットが付属しています。

+ MNIST 手書き数字データベース
+ Fashion-MNIST ファッション記事データベース
+ CIFAR10 画像分類
+ CIFAR100 画像分類

> Fashion-MNIST ファッション記事データベースはKeras 2.0.8にはバンドルされていません。

また画像認識用途ではありませんが、以下のテキストデータセットも付属されています。

+ IMDB映画レビュー感情分類
+ ロイターのニュースワイヤー トピックス分類
+ ボストンの住宅価格回帰データセット

<div style="page-break-before:always"></div>

## MNIST 手書き数字データベース

60,000枚の28x28、10個の数字の白黒画像と10,000枚のテスト用画像データセット。

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Fashion-MNIST ファッション記事データベース

60,000枚の28x28、10個のファッションカテゴリの白黒画像と10,000枚のテスト用画像データセット。

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

> Fashion-MNIST ファッション記事データベースはKeras 2.0.9からバンドルされています。

## CIFAR10 画像分類

10のクラスにラベル付けされた、50000枚の32x32訓練用カラー画像、10000枚のテスト用画像のデータセット。

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```


## CIFAR100 画像分類

100のクラスにラベル付けされた、50000枚の32x32訓練用カラー画像、10000枚のテスト用画像のデータセット。

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

<div style="page-break-before:always"></div>

## IMDB映画レビュー感情分類

感情 (肯定/否定) のラベル付けをされた、25,000のIMDB映画レビューのデータセット。

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```

## ロイターのニュースワイヤー トピックス分類

46のトピックにラベル付けされた、11228個のロイターのニュースワイヤーのデータセット。

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

## ボストンの住宅価格回帰データセット

Carnegie Mellon大学のStatLib ライブラリのデータセット。

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```
