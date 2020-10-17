import time
import numpy as np


class NNet:
    def __init__(self):
        self._sequence = []
        self._weights = []

    def __repr__(self):
        rep = str(self.sequence[0])
        for layer in self.sequence[1:]:
            rep += '-' + str(layer)
        return "NNet model {}".format(rep)

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, seq):
        if not isinstance(seq, list):
            raise TypeError("sequence input should be a list")
        self._sequence = seq
        self._num_layers = len(seq) + 1
    
    @property
    def weights(self):
        return self._weights

    def diff_tanh(self, s):
        return 1-np.tanh(s)**2

    def fit(self, train_X, train_Y, weight_range, learning_rate=0.1,T=500):
        dim, num_data = train_X.shape[1], train_X.shape[0]
        X = np.hstack((np.array([[1.]*num_data]).T, train_X))
        
        # 初始化權重陣列
        weights = []
        num_neurons = [dim] + self._sequence
        for l in range(1, self._num_layers):
            # 權重的第二維度等於 (l+1)層 neuron 的數量 (沒有 bias)
            # 權重的第一維度等於 (l)層 neuron 的數量 (包含 bias)
            weight = np.random.uniform(
                     weight_range[0], weight_range[1], 
                     (num_neurons[l], num_neurons[l-1]+1)
                    )
            weights.append(weight)
        
        for _ in range(T):
            # 隨機選擇資料點
            i = np.random.randint(0, num_data-1)
            scores, neurons = self.forward(X[i], weights)
            self.backprop(train_Y[i], scores, neurons, weights, learning_rate)

        self._weights = weights

    def forward(self, Xn, weights):
        # 初始化每一層的神經元, scores 紀錄神經元的權重加總 (不包含第一層)
        scores = [
            np.array([0.]*n) for n in self._sequence[:-1]
        ] + [np.array([0.]*self._sequence[-1])]

        # neurons 紀錄神經元 tanh 轉換後的數值 (不包含最後一層)
        neurons = [Xn] + [
            np.array([1.]+[0.]*n) for n in self._sequence[:-1]
        ]

        # 向前傳播
        for i in range(1, self._num_layers-1):
            scores[i-1] = np.dot(weights[i-1], neurons[i-1])
            neurons[i][1:] = np.tanh(scores[i-1])
        
        # 傳播至最後一層
        scores[-1] = np.dot(weights[-1], neurons[-1])
        return scores, neurons
    
    def backprop(self, Yn, scores, neurons, weights, learning_rate):
        # 初始化 error 對分數偏導數的 List，並計算最後一層的偏導
        delta = [np.array(0)] * (self._num_layers-1)
        delta[-1] = -2*(Yn-scores[-1])

        # 往後傳播計算 error 對各層分數的偏導數
        for i in range(1, self._num_layers-1):
            delta[-(i+1)] = delta[-i]@weights[-i][:,1:]*self.diff_tanh(scores[-(i+1)])

        # 權重更新
        for i in range(self._num_layers-1):
            weights[i] = weights[i] - learning_rate*neurons[i]*delta[i][:, np.newaxis]
    
    def predict(self, test_X, test_Y):
        num_data = test_X.shape[0]
        X = np.hstack((np.array([[1.]*num_data]).T, test_X))

        error = 0
        for Xn, Yn in zip(X, test_Y):
            # 向前傳播取的最終預測值
            scores, _ = self.forward(Xn, self._weights)
            pred = scores[-1]
            
            if np.sign(pred) != Yn:
                error += 1

        return error/num_data

def load_data(file_path):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            vals = line.split()
            x = list(map(lambda x: float(x), vals[:-1]))
            y = int(vals[-1])
            
            X.append(x)
            Y.append(y)
        
        X = np.array(X)
        Y = np.array(Y)
    return X, Y

if __name__ == "__main__":
    # Q11
    N = 5 # 題目要求 500 次， run 5 次節省時間
    T = 50000
    learning_rate = 0.1
    weight_range = (-0.1, 0.1)
    hidden_neuron = [1, 6, 11, 16, 21]

    train_data = "hw4_nnet_train.txt"
    test_data = "hw4_nnet_test.txt"
    train_X, train_Y = load_data(train_data)
    test_X, test_Y = load_data(test_data)
    nn = NNet()

    error_M = []
    for hidden in hidden_neuron:
        error = 0
        for _ in range(N):
            nn.sequence = [hidden]+[1]
            nn.fit(train_X, train_Y, weight_range, learning_rate, T)
            error += nn.predict(test_X, test_Y)

        error_M.append(error/N)

    print("Q11: {}".format(error_M))

    # Q12
    error_r = []
    weight_ranges = [(0, 0), (-0.001, 0.001), (-0.1, 0.1), (-10, 10), (-1000, 1000)]
    for weight_range in weight_ranges:
        error = 0
        for _ in range(N):
            nn.sequence = [3]+[1]
            nn.fit(train_X, train_Y, weight_range, learning_rate, T)
            error += nn.predict(test_X, test_Y)

        error_r.append(error/N)

    print("Q12: {}".format(error_r))

    # Q13
    error_l = []
    rates = [0.001, 0.01, 0.1, 1, 10]
    for learning_rate in rates:
        error = 0
        for _ in range(N):
            nn.sequence = [3]+[1]
            nn.fit(train_X, train_Y, weight_range, learning_rate, T)
            error += nn.predict(test_X, test_Y)

        error_l.append(error/N)

    print("Q13: {}".format(error_l))   

    # Q14
    learning_rate = 0.1
    weight_range = (-0.1, 0.1)
    nn = NNet()

    error_d = 0
    for _ in range(N):
        nn.sequence = [8, 3]+[1]
        nn.fit(train_X, train_Y, weight_range, learning_rate, T)
        error_d += nn.predict(test_X, test_Y)

    print("Q14: {}".format(error_d/N))   