import numpy as np

def load_data(file_path, split=400):
    with open(file_path, 'r') as file:
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []

        lines = file.readlines()
        train_lines = lines[:400]
        test_lines = lines[400:]
        for line in train_lines:
            vals = line.split()
            x = list(map(float, vals[:-1]))
            y = int(vals[-1])

            train_X.append(x)
            train_Y.append(y)
        
        for line in test_lines:
            vals = line.split()
            x = list(map(float, vals[:-1]))
            y = int(vals[-1])

            test_X.append(x)
            test_Y.append(y)

        return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

class LSSVM:
    
    def RBF_kernel(self, x1, x2, gamma):
        return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))
    
    def kernel_matrix(self, X1, X2, gamma):
        N1, _ = X1.shape
        N2, _ = X2.shape
        K = np.zeros((N1, N2))
        for i in range(N1):
            for j in range(N2):
                K[i][j] = self.RBF_kernel(X1[i], X2[j], gamma)
        
        return K
    
    def beta(self, lamb, K, Y):
        N = len(Y)
        inv_matrix = np.linalg.pinv(lamb*np.identity(N) + K)
        return inv_matrix @ Y
    
    def valid(self, Y, K, beta):
        predict_Y = np.sign(K.T.dot(beta))
        error = sum(predict_Y!=Y)
        return error/len(Y)

if __name__ == '__main__':
    file_path = "hw2_lssvm_all.txt"
    train_X, train_Y, test_X, test_Y = load_data(file_path)
    gamma = [32, 2, 0.125]
    lamb = [0.001, 1, 1000]
    lssvm = LSSVM()

    E_in = []
    E_out = []
    combine = [(g, l) for g in gamma for l in lamb]
    for (g, l) in combine:
        K1 = lssvm.kernel_matrix(train_X, train_X, g)
        K2 = lssvm.kernel_matrix(train_X, test_X, g)
        beta = lssvm.beta(l, K1, train_Y)

        e_in = lssvm.valid(train_Y, K1, beta)
        e_out = lssvm.valid(test_Y, K2, beta)

        E_in.append(e_in)
        E_out.append(e_out)

    # Q19    
    print(E_in)
    # Q20
    print(E_out)
