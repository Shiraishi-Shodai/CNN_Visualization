import torch
from utils import im2col, col2im, softmax, cross_entropy_error

class ReLU():
    
    def __init__(self):
        self.x = None
        self.params = []
        self.grads = []
    
    def forward(self, x):
        """
        """
        self.x = x
        out = torch.where(self.x > 0, self.x, 0)
        # print(f"ReLUで0になる割合 : {(len(out.flatten()) - torch.count_nonzero(out).item()) / len(out.flatten()) * 100}%")
        return out
    
    def backward(self, dout=1):
        return torch.where(self.x > 0, dout, 0)

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [torch.zeros_like(W), torch.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = x @ W + b
        self.x = x
        
        return out

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        db = torch.sum(dout, dim=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        """
        Parameters
        ----------
        x : 実数を持つ行列(全結合の出力結果)
        t : one-hotベクトルの多値分類結果
            
        Returns
        -------
        loss : 損失(スカラ)
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss.detach().item()
    
    def backward(self, dout=1):
        """
        Parameters
        ----------
        dout : 1
        
        
        Returns
        -------
        dx : softmaxへの入力xの勾配
             y - tを行うことでdxは0か負の実数値になる。
             勾配降下法では、new_w = -now_w * η * (dL / dW)となる
             dWが正の値であれば、new_wはwよりも小さくなる(減少する) → そのニューロンの出力を小さくしようとする
             dWが負の値であれば、new_wはwよりも大きくなる(増加する) → そのニューロンの出力を大きくしようとする
             
             正解ラベルの出力値を予測するニューロンの重みを大きくしたいから、y - tで正解ラベルの勾配dxが0か負の実数値になるようにする。
             そうすることで、dx = 0なら完璧な予測だから更新なし。負の値なら次回予測時に正解ラベルの予測値を大きくしにかかる
        """
        
        N, D = self.y.shape
        
        if self.t.size() == self.y.size():
            # t : one-hotベクトルの行列想定
            dx = dout * (self.y - self.t) / N
        else:
            # t : ラベルのベクトル想定
            dx = self.y.clone()
            dx[torch.arange(N), self.t] -= 1
            dx = dout * dx / N
        
        return dx
        
class Convolution:
    def __init__(self, weight, bias, stride=1, pad=0):
        self.params = [weight, bias]
        self.grads = [torch.zeros_like(weight), torch.zeros_like(bias)]
        self.stride = stride
        self.pad = pad
        
        # 逆伝搬用データ
        self.x = None
        self.col = None
        self.col_W = None
        
        # 重み
        self.dW = None
        self.db = None
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        
        Returns
        -------
        out : 4次元配列(N, out_h, out_w, 1)
        """
        weight, bias = self.params
        FN, C, FH, FW = weight.shape
        N, C, H, W = x.shape
        out_h = (H + self.pad * 2 - FH) // self.stride + 1
        out_w = (W + self.pad * 2 - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)             # (batch_size * out_h * out_w, C * FH * FW)
        col_W = weight.reshape(FN, -1).T                           #(FN, C * FH * FW) → (C * FH * FW, FN)
        # (受容野, カーネルの各係数にかける入力ピクセル) @ (カーネルの各係数, 出力チャンネル) = (受容野, 出力チャンネル)
        out = col @ col_W + bias                                 # (batch_size * out_h * out_w, C * FH * FW) @ (C * FH * FW, FN) = (batch_size * out_h * out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2) # (batch_size * out_h * out_w, FN) → (batch_size, FN, out_h, out_w)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        # print(f"mean     : {out.mean():.6f}")
        # print(f"abs mean : {out.abs().mean():.6f}")
        # print(f"std      : {out.std():.6f}")
        # print(f"max      : {out.max():.6f}")
        # print(f"min      : {out.min():.6f}")
        return out
    
    def backward(self, dout):
        weight, bias = self.params
        FN, C, FH, FW = weight.shape
        # dout (N, out_channel, out_h, out_w)
        # (N, out_channel, out_h, out_w) → (N, out_h, out_w, out_channel) → (N * out_h * out_w, out_channel) 
        # (勾配受容野, 出力チャンネル)
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)
        # 各受容野の勾配の合計を出力チャンネルごとに計算
        db = torch.sum(dout, dim=0)
        # (カーネルの各係数にかける入力ピクセル, 受容野) @ (勾配受容野, 出力チャンネル) = (カーネルの各係数にかける入力ピクセル, 出力チャンネル)
        dW = self.col.T @ dout
        dW = dW.permute(1, 0).reshape(FN, C, FH, FW)
        # (勾配受容野, 出力チャンネル) @ (出力チャンネル, カーネルの各係数) = (勾配受容野, カーネルの各係数)
        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx
    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.params = []
        self.grads = []
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # (batch_size * out_h * out_w, C * pool_h * pool_w)
        col = col.reshape(-1, self.pool_h*self.pool_w) # (batch_size * out_h * out_w * C, pool_h * pool_w)

        arg_max = torch.argmax(col, dim=1) # プーリング領域での最大値をのインデックスをとる
        out = torch.max(col, dim=1).values # プーリング領域の最大値を求める
        out = out.reshape(N, out_h, out_w, C).permute(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        
        return out
        
    def backward(self, dout):
        dout = dout.permute(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = torch.zeros((dout.numel(), pool_size), device=dout.device, dtype=dout.dtype)
        dmax[torch.arange(self.arg_max.numel()), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
    
class Flatten:
    def __init__(self):
        self.params = []
        self.grads = []
        self.original_shape = None
    
    def forward(self, x):
        """フラットにする
        Parameter
        ---------
        x : (N, C, H, W)
        
        Returns
        -------
        out : (N, C * H * W)
        """
        
        self.original_shape = x.shape
        return x.reshape(self.original_shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.original_shape)
        

class Dropout:
    def __init__(self, dropout_rate):
        """
        Parameter
        --------
        dropout_rate : ニューロンの接続を切る(dropoutする割合)
        """
        
        self.params = []
        self.grads = []
        self.dropout_rate = dropout_rate
        self.mask = None
        self.train = True
    
    def forward(self, x):
        
        if self.train:
            self.mask = torch.rand_like(x) > self.dropout_rate
            return (x * self.mask) / (1 - self.dropout_rate)
        else:
            return x
    
    def backward(self, dout):
        return (dout * self.mask) / (1 - self.dropout_rate)

class BatchNorm:
    """チャネルごとに平均する。各チャネルにそれぞれ抽出している特徴を持つため
    """
    def __init__(self, gamma, beta, epsilon, momentum):
        """
        Parameters
        -----------
        gamma: (C,)
        beta : (C,)
        """
        self.params = [gamma, beta]
        self.grads = [torch.zeros_like(gamma), torch.zeros_like(beta)]
        self.epsilon = epsilon
        self.momentum = momentum
        self.train = True
        self.cache = None
        self.running_mean = None
        self.running_var = None
    
    def _get_reduce_dims(self, x):
        """標準化する軸を返す
        """
        if x.ndim == 2:
            return (0,)
        elif x.ndim == 4:
            return (0, 2, 3)
        else:
            raise ValueError(f"xの次元を確認してね: {x.shape}")
    
    def _get_n(self, x, dims):
        """平均に使用するNを計算。入力xによって値が変わるため
        """
        n = 1
        
        for dim in dims:
            n *= x.shape[dim]
        
        return n

    
    def _get_running_shape(self, x, dims):
        """バッチごとの平均や標準偏差を格納するテンソルの次元を返す
        """
        running_shape = tuple([1 if i in dims else x.shape[i] for i in range(x.ndim)])
        return running_shape

    def forward(self, x):
        """
        BatchNormは各特徴量ごとに標準化を行う
        Affineであれば(N, D) → N方向に標準化
        Convolutionであれば(N, C, H, W) → (N, H, W)方向に標準化
        Parameters
        -----------
        x: (N, C, H, W)
        
        Returns
        ----------
        x_normalized: (N, C, H, W)
        """
        gamma, beta = self.params

        out = None
        dims = self._get_reduce_dims(x) # 標準化する軸を取得
        running_shape = self._get_running_shape(x, dims)
        
        if self.running_mean is None:
            self.running_mean = torch.zeros(running_shape, dtype=x.dtype, device=x.device)

        if self.running_var is None:
            self.running_var = torch.ones(running_shape, dtype=x.dtype, device=x.device)
        
        match self.train:
            case True:
                mean = x.mean(dim=dims, keepdim=True)
                variance = x.var(dim=dims, keepdim=True, correction=0)
                std = torch.sqrt(variance + self.epsilon)
                x_hat = (x - mean) / std # (x - mean) * ( 1 / std)
                # バッチごとの移動平均を更新(Epochにまたがる)
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance
                    
                self.cache = (x, mean, variance, x_hat, std)
                out = gamma * x_hat + beta # ブロードキャスト問題、gammaとbetaのshapeを整える

            case False:
                
                out = gamma * (x - self.running_mean) / (torch.sqrt(self.running_var)) + beta
            case _:
                raise ValueError(f"BatchNormのモードがおかしいよ→ {self.train}")
        
        return out
    
    def backward(self, dout):
        """逆伝搬
        
        Parameters
        -----------
        dout : 前の層の勾配
        
        Returns
        -----------
        dx   : 入力に対する勾配
        """
        
        dx, dgamma, dbeta = None, None, None
        x, mean, variance, x_hat, std = self.cache
        gamma, beta = self.params 
        
        dims = self._get_reduce_dims(x) # 標準化する軸を取得
        N = 1.0 * self._get_n(x, dims) # 平均に使用するNを取得

        dgamma = torch.sum(dout * x_hat, dim=dims, keepdim=True)
        dbeta = torch.sum(dout, dim=dims, keepdim=True)

        dydx_hat = dout * gamma
        dx_hatdμ = -1 / std # x_hatに対するμについての微分
        dx_hatdv = -0.5 * (variance ** (-1.5)) * (x - mean) # x_hatに対する分散についての微分 (-1.5 = -3/2)
        dx_hatdx = 1 / std # x_hatに対するxについての微分(μとσは定数として扱う)
        dvdx = 2 / N * (x - mean) # 分散に対するxについての微分
        dvdu = -2 / N * torch.sum(x - mean, dim=dims, keepdim=True) # 分散に対するμについての微分
        dudx = torch.ones_like(x) / N # μに対するxについての微分
        
        # doutに対するxの微分(doutに対するxの直接経路 + doutに対する平均を経由する経路 + doutに対する分散を経由する経路)
        # 複数の経路がある場合は、すべての経路の勾配を足し算する

        dx = dydx_hat * dx_hatdx + \
            torch.sum(dydx_hat * dx_hatdμ, dim=dims, keepdim=True) * dudx + \
            torch.sum(dydx_hat * dx_hatdv, dim=dims, keepdim=True) * (dvdx + dvdu * dudx)
        
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx