import numpy as np
from copy import deepcopy

T = 300
class AdaBoost_Stump:
    def __init__(self, file_path):
        self.train_X, self.train_Y = AdaBoost_Stump.load_data(file_path)
        self.row_size, self.col_size = self.train_X.shape

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r') as file:
            X = []
            Y = []
            lines = file.readlines()
            for line in lines:
                vals = line.split()
                x = [float(v) for v in vals[:-1]]
                y = int(vals[-1])
                X.append(x)
                Y.append(y)
            X = np.array(X)
            X = X.T
        return X, np.array(Y)

    def decision_stump(self, X, Y, u):
        s = 1
        best_theta = 0
        best_index = 0
        min_error = float("inf")

        for i in range(self.row_size):
            sort_index = np.argsort(X[i])
            sort_X = X[i][sort_index]
            sort_Y = Y[sort_index]
            sort_u = u[sort_index]

            thetas = np.array([float("-inf")] + [(sort_X[i]+sort_X[i+1])/2 for i in range(self.col_size-1)])
            for theta in thetas:
                y_positive = np.where(sort_X > theta, 1, -1)
                y_negative = np.where(sort_X < theta, 1, -1)

                positive_error = np.sum(sort_u[y_positive!=sort_Y])
                negative_error = np.sum(sort_u[y_negative!=sort_Y])

                if positive_error < min_error:
                    min_error = positive_error
                    best_theta = theta
                    best_index = i
                    s = 1
                
                if negative_error < min_error:
                    min_error = negative_error
                    best_theta = theta
                    best_index = i
                    s = -1

        return min_error, best_theta, s, best_index

    def sign(self, x, theta, s=1):
        return s*1 if x-theta > 0 else s*(-1)

    def compute_alpha(self, min_error, u):
        error_rate = min_error/np.sum(u)
        t_weight = np.sqrt((1-error_rate)/error_rate)
        return t_weight, np.log(t_weight)
        
    def train(self, T):
        X = deepcopy(self.train_X)
        Y = deepcopy(self.train_Y)
	
	# 初始化錯誤權重 u_t
        u = np.ones(self.col_size)/self.col_size
        u_sign = np.vectorize(self.sign)

        u_t = []
        u_t.append(deepcopy(u))
        epsilon_t = []
        alphas = []
        hypotheses = []
        for t in range(T):
            # 取得 u_t 對應的最佳 decision-stump 參數，並計算更新的 u_t 權重，以及 hypothese 線性組合的權重
            error, theta, s, best_index = self.decision_stump(X, Y, u)
            t_weight, alpha = self.compute_alpha(error, u)

            epsilon_t.append(error/np.sum(u))
            alphas.append(alpha)
            hypotheses.append([s, theta, best_index])
	    
	    # 依錯誤率及公式 "sqrt((1-error_rate)/error_rate)" 更新錯誤權重 u_t
            u[u_sign(X[best_index], theta, s)!=Y] *= t_weight
            u[u_sign(X[best_index], theta, s)==Y] /= t_weight
            u_t.append(deepcopy(u))

        self.u_t = u_t
        self.epsilon_t = epsilon_t
        self.alphas = alphas
        self.hypotheses = hypotheses
    
    def valid(self, X, Y):
        u_sign = np.vectorize(self.sign)

        score_array = np.zeros_like(X[0])
        for alpha, (s, theta, best_index) in zip(self.alphas, self.hypotheses):
            score_array += alpha * u_sign(X[best_index], theta, s)

            predict = np.sign(score_array)
            error = np.sum(predict!=Y)
        
        return error/X.shape[1]

if __name__ == '__main__':
    train_file = "hw2_adaboost_train.txt"
    test_file = "hw2_adaboost_test.txt"
    adaBoost = AdaBoost_Stump(train_file)
    test_X, test_Y = AdaBoost_Stump.load_data(test_file)

    # Q12
    u0 = np.array([1/adaBoost.col_size for _ in range(adaBoost.col_size)])
    error, theta, s, best_index = adaBoost.decision_stump(adaBoost.train_X, adaBoost.train_Y, u0)
    print("Q12:", error)
    # Q17
    adaBoost.train(1)
    Eout_1 = adaBoost.valid(test_X, test_Y)
    print("Q17:", Eout_1)
    # Q13
    adaBoost.train(T)
    error_rate = adaBoost.valid(adaBoost.train_X, adaBoost.train_Y)
    print("Q13:", error_rate)
    # Q18
    Eout_G = adaBoost.valid(test_X, test_Y)
    print("Q18:", Eout_G)
    # Q14
    print("Q14:", np.sum(adaBoost.u_t[1]))
    # Q15
    print("Q15:", np.sum(adaBoost.u_t[300]))
    # Q16
    print("Q16:", min(adaBoost.epsilon_t))
