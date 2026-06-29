ありがとうございます。
この方法は**PyTorchのHookとほぼ同じ思想**なので、今後PyTorchのソースコードを読む際にも役立ちます。

---

# 全体の流れ

```text
Trainer
    │
    │ hookを登録
    ▼
Model.forward(x, hook)

    Conv1.forward()
          │
          ▼
    hook(layer, output)

    Conv2.forward()
          │
          ▼
    hook(layer, output)

    Pool.forward()
          │
          ▼
    hook(layer, output)
```

Modelは

> 「今このレイヤーの出力ができました！」

と通知するだけです。

---

# ① Hookのインターフェース

まずは一番単純な形です。

```python
def hook(layer, output):
    ...
```

引数は

* layer
* output

だけです。

例えば

```python
def print_hook(layer, output):
    print(layer.name, output.shape)
```

---

# ② Model側

Modelはhookを受け取ります。

```python
class Sequential:

    def forward(self, x, hook=None):

        for layer in self.layers:

            x = layer.forward(x)

            if hook is not None:
                hook(layer, x)

        return x
```

これだけです。

Modelは

* 保存
* ログ
* 可視化

を知りません。

---

# ③ Trainer側

Trainerでは

```python
recorder = Recorder(max_samples=5)

model.forward(
    x,
    hook=recorder.forward_hook
)
```

とします。

---

# ④ Recorder

Recorderは

```python
class Recorder:

    def __init__(self):
        self.outputs = {}

    def forward_hook(self, layer, output):

        self.outputs[layer.name] = output.detach().clone()
```

これで

```text
outputs

Conv1
    ↓
Tensor(...)

Conv2
    ↓
Tensor(...)

Pool1
    ↓
Tensor(...)
```

となります。

---

# ⑤ 逆伝搬

全く同じです。

```python
def backward(self, dout, hook=None):

    for layer in reversed(self.layers):

        dout = layer.backward(dout)

        if hook is not None:
            hook(layer, dout)

    return dout
```

Recorderは

```python
class Recorder:

    def backward_hook(self, layer, grad):

        self.grads[layer.name] = grad.detach().clone()
```

---

# ⑥ ForwardとBackwardを分ける

私はこうします。

```python
class Recorder:

    def forward_hook(self, layer, output):
        ...

    def backward_hook(self, layer, grad):
        ...
```

Trainerでは

```python
model.forward(
    x,
    hook=self.recorder.forward_hook
)

...

model.backward(
    dout,
    hook=self.recorder.backward_hook
)
```

非常に分かりやすいです。

---

# ⑦ HookManagerを作るとさらに拡張しやすい

PyTorchは一つのHookしか登録できるわけではありません。

例えば

```text
Conv1

↓

Recorder

↓

Logger

↓

Debugger
```

のように複数同時に呼べます。

そこで

```python
class HookManager:

    def __init__(self):
        self.forward_hooks = []

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)

    def call_forward_hooks(self, layer, output):

        for hook in self.forward_hooks:
            hook(layer, output)
```

Modelは

```python
class Sequential:

    def __init__(self):
        self.hooks = HookManager()

    def forward(self, x):

        for layer in self.layers:

            x = layer.forward(x)

            self.hooks.call_forward_hooks(
                layer,
                x
            )

        return x
```

Trainerは

```python
model.hooks.register_forward_hook(
    recorder.forward_hook
)

model.hooks.register_forward_hook(
    logger.forward_hook
)
```

こうすると

```text
Conv1
    │
    ├── Recorder
    ├── Logger
    ├── Visualizer
    └── Debugger
```

のように後から機能を追加してもModelを一切変更する必要がありません。

---

## 私ならこの構成を採用します

あなたは将来的に

* CNNの特徴マップ表示
* 勾配のヒートマップ
* Webアプリで各レイヤーをクリックして可視化
* 学習ログの保存

まで考えています。

そのため、**ModelはHookを呼び出すだけ**にして、`Recorder`、`Logger`、`Visualizer`などをHookとして登録できる設計にしておくと、機能追加のたびにModelを書き換える必要がなくなります。これは拡張性が高く、実際のフレームワークにも近い設計です。
