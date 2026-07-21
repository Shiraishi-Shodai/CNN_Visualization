```python
import torch
from copy import deepcopy

W = torch.tensor([1.0])

params = [W]
backup = deepcopy(params)

params[0] += 3


print(W)
print(params[0])

params[0].copy_(backup[0])
# params[0] = backup[0]

print(W)
print(params[0])
```

もちろんです。この問題は**Pythonの「変数」と「オブジェクトの参照」**を理解すると納得できます。

---

# まず前提

例えばモデルが

```python
class Linear:
    def __init__(self):
        self.W = torch.tensor([1.0])

layer = Linear()

params = [layer.W]
```

だったとします。

このとき

```python
layer.W
```

と

```python
params[0]
```

は**同じTensor**を指しています。

イメージすると

```text
        ┌────────────┐
layer.W │ Tensor(1.0)│
        └────────────┘
              ▲
              │
        params[0]
```

---

# lookahead後

例えば

```python
params[0] += 3
```

すると

```text
        ┌────────────┐
layer.W │ Tensor(4.0)│
        └────────────┘
              ▲
              │
        params[0]
```

になります。

両方とも4になります。

---

# restoreで今のコードを書く

今のコードは

```python
params[0] = backup_params[0]
```

です。

すると

```text
backup_params[0]
        │
        ▼
   ┌────────────┐
   │ Tensor(1.0)│
   └────────────┘

params[0]
        │
        ▼
   ┌────────────┐
   │ Tensor(1.0)│
   └────────────┘


layer.W
        │
        ▼
   ┌────────────┐
   │ Tensor(4.0)│
   └────────────┘
```

ここで重要なのは

**paramsだけが別のTensorを指すようになった**

ということです。

---

# 実際にコードで見る

```python
import torch
from copy import deepcopy

W = torch.tensor([1.0])

params = [W]
backup = deepcopy(params)

params[0] += 3

print(W)
print(params[0])
```

出力

```text
tensor([4.])
tensor([4.])
```

ここまではOKです。

---

次に

```python
params[0] = backup[0]
```

を実行します。

```python
print(W)
print(params[0])
```

すると

```text
tensor([4.])
tensor([1.])
```

になります。

**paramsは戻ったのに、Wは戻っていません。**

---

# なぜ？

代入

```python
params[0] = backup[0]
```

は

Tensorの中身を書き換えているのではありません。

**params[0]が指す相手を変えただけ**です。

つまり

```text
変更前

params[0]
      │
      ▼
Tensor(4)

変更後

params[0]
      │
      ▼
Tensor(1)
```

になっただけです。

---

# copy_を使うと？

今度は

```python
params[0].copy_(backup[0])
```

を使います。

すると

```text
        ┌────────────┐
layer.W │ Tensor(4.0)│
        └────────────┘
              ▲
              │
        params[0]
```

という**同じTensor**の中身だけを

```text
4
```

↓

```text
1
```

に書き換えます。

つまり

```text
        ┌────────────┐
layer.W │ Tensor(1.0)│
        └────────────┘
              ▲
              │
        params[0]
```

になります。

参照は変わっていません。

---

# 実際のコード

```python
import torch
from copy import deepcopy

W = torch.tensor([1.0])

params = [W]
backup = deepcopy(params)

params[0] += 3

params[0].copy_(backup[0])

print(W)
print(params[0])
```

出力

```text
tensor([1.])
tensor([1.])
```

両方とも元に戻ります。

---

# あなたのフレームワークではどうか？

ここで一つ重要な確認があります。

あなたのフレームワークでは、`params` は

```python
[
    layer1.W,
    layer1.b,
    layer2.W,
    ...
]
```

のように**各レイヤーのパラメータそのものへの参照を集めたリスト**になっていますか？

もしそうであれば、**`params[i] = ...` は危険**で、`copy_()` を使う必要があります。

一方で、もし毎回 `params` を新しく生成し、レイヤー側はそのリストを一切保持していない特殊な設計であれば、`params[i] = ...` でも動作する可能性があります。

この点が分かれば、あなたの実装に最適な復元方法を判断できます。