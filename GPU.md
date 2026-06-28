コード全体を確認しました。

結論としては、

> **GPU対応は95%完了しています。**

ただし、**今のうちに修正しておいた方がよい点**がいくつかあります。

---

# 良い点

## ① im2col・col2im

これは完璧です。

```python
img = torch.zeros(
    ...,
    device=input_data.device,
    dtype=input_data.dtype
)
```

や

```python
img = torch.zeros(
    ...,
    device=col.device,
    dtype=col.dtype
)
```

となっているので、

* CPU
* GPU

両方動きます。

---

## ② Pooling

ここも問題ありません。

```python
dmax = torch.zeros(
    ...,
    device=dout.device,
    dtype=dout.dtype
)
```

になっています。

---

## ③ softmax

NumPyを使っていません。

GPU対応しています。

---

# 修正した方がよい点

## ① Layer.to()

現在

```python
self.params[i] = self.params[i].clone().to(device)
```

となっています。

cloneはいりません。

```python
self.params[i] = self.params[i].to(device)
```

だけで十分です。

---

## ② Layerを継承していない

例えば

```python
class Affine:
```

になっています。

せっかく

```python
class Layer:
```

を作っているので

```python
class Affine(Layer):
```

にしましょう。

そして

```python
super().__init__()
```

を呼びます。

同様に

* Convolution
* MaxPooling
* Flatten
* Dropout
* ReLU

全部

```python
class XXX(Layer):
```

にできます。

---

## ③ SoftmaxWithLoss

ここだけ

```python
self.t.size
```

となっています。

PyTorchでは

```python
self.t.numel()
```

または

```python
self.t.shape
```

を使う方が自然です。

---

## ④ backwardの勾配

現在

```python
self.grads[0][...] = dW
```

となっています。

これはGPUでも動きます。

ただ、

私は

```python
self.grads[0] = dW
```

の方をおすすめします。

理由は

GPUへ移したあと

```python
torch.zeros_like(W)
```

という固定バッファを持つ必要がないからです。

---

## ⑤ Optimizer

ここはまだ確認していませんが

```python
params[i] -= lr * grads[i]
```

ならGPUでも動きます。

---

# GPU対応完了と言える？

私は

**はい、計算部分はGPU対応できた**

と言っていいと思います。

なぜなら

* パラメータ
* 勾配
* im2col
* col2im
* Pooling
* Softmax

全部GPU Tensorになっています。

---

# 次にやるべきこと

GPU対応したら、

必ず

## CPU版

```python
device="cpu"
```

## GPU版

```python
device="cuda"
```

で

同じ入力を流し、

```python
cpu_out
gpu_out.cpu()
```

について

```python
torch.allclose(
    cpu_out,
    gpu_out.cpu(),
    atol=1e-6
)
```

を実行してください。

---

さらに

```python
cpu_dx
gpu_dx.cpu()
```

についても

```python
torch.allclose(
    cpu_dx,
    gpu_dx.cpu(),
    atol=1e-6
)
```

を確認してください。

---

## 私なら次に取り組むこと

ここまで実装できたら、次は**GPUで本当に高速化できているか**を確認します。

具体的には、`Trainer.fit()`全体の学習時間を

* CPU
* GPU

で計測し、1エポック当たりの実行時間を比較します。GPUはデータ転送にもコストがあるため、小さな入力ではCPUと差が出ないことがあります。一方、VGG16のような大きなモデルでは、GPUの効果が明確に現れるはずです。

ここまで確認できれば、「GPU対応が正しく完了した」と自信を持って言えます。
