import numpy as np

class K_NN:
    def __init__(self):
        self._k_pos = None
        self._k_centers = None

    @property
    def k_pos(self):
        return self._k_pos

    @property
    def k_centers(self):
        return self._k_centers

    def fit(self, train_X, train_Y, test_X, K):
        # 擴展維度，以適用向量化計算
        num_data, dim = train_X.shape[0], train_X.shape[1]
        X = np.tile(test_X, num_data).reshape(-1, num_data, dim)

        # 計算向量間的 l2_norm
        diff = X-train_X
        l2_norm = np.sum(np.square(diff), axis=2)
        sort_pos = np.argsort(l2_norm, axis=1)

        # 並取出前 k　個中心點位置 (包含自己)
        self._k_pos = sort_pos[:, :K]
        self._k_centers = train_X[self._k_pos]
    
    def predict(self, train_X, train_Y, test_X, test_Y, K):
        # fit 取得 K-NN 並預測
        self.fit(train_X, train_Y, test_X, K)
        Y_pred = np.sign(np.sum(train_Y[self._k_pos], axis=1))

        # 計算錯誤率
        error = sum(Y_pred!=test_Y)
        error = error/test_X.shape[0]

        return error

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
    train_data = "hw4_nbor_train.txt"
    test_data = "hw4_nbor_test.txt"
    train_X, train_Y = load_data(train_data)
    test_X, test_Y = load_data(test_data)
    
    # Q15
    K = 1
    k_nn = K_NN()
    error = k_nn.predict(train_X, train_Y, train_X, train_Y, K)
    print("Q15: {}".format(error))

    # Q16
    K = 1
    error = k_nn.predict(train_X, train_Y, test_X, test_Y, K)
    print("Q16: {}".format(error))

    # Q17
    K = 5
    k_nn = K_NN()
    error = k_nn.predict(train_X, train_Y, train_X, train_Y, K)
    print("Q17: {}".format(error))

    # Q18
    K = 5
    error = k_nn.predict(train_X, train_Y, test_X, test_Y, K)
    print("Q18: {}".format(error))