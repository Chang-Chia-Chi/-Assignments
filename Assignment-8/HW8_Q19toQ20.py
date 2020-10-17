import numpy as np


class K_Means:
    def __init__(self):
        self._k_centers = None

    @property
    def k_centers(self):
        return self._k_centers
    
    def k_means(self, train_X, K):
        # 隨機取 k 個點當作中心
        num_data = train_X.shape[0]
        center_i = np.random.randint(0, num_data-1, K)
        k_centers = train_X[center_i]

        # 先執行一次分類
        X = np.tile(train_X, K).reshape(num_data, K, -1)
        diff = X - k_centers
        
        l2_norm = np.sum(np.square(diff), axis=2)
        sort_pos = np.argsort(l2_norm, axis=1)[:,0]
        
        curr_clus = [train_X[np.where(sort_pos==i)[0]] for i in range(K)]
        prev_clus = []
        
        # 開始 alternating optimization
        while True:
            # 利用 cluster 計算新的中心點 (注意 cluster 有可能完全沒有東西)
            k_centers = [
                np.mean(curr_clus[i], axis=0) 
                if len(curr_clus[i]) != 0 else k_centers[i] 
                for i in range(K)
            ]

            # 重新分類
            diff = X - k_centers
            l2_norm = np.sum(np.square(diff), axis=2)
            sort_pos = np.argsort(l2_norm, axis=1)[:,0]

            prev_clus = curr_clus
            curr_clus = [train_X[np.where(sort_pos==i)[0]] for i in range(K)]

            if all(np.array_equal(prev_clus[i], curr_clus[i]) for i in range(K)):
                break
                
        self._k_centers = k_centers

        # 計算錯誤
        diff = [curr_clus[i] - k_centers[i] for i in range(K)]
        error = [sum(np.sum(diff[i]**2, axis=1)) for i in range(K)]
        error = sum(error)/num_data

        return error

def load_data(file_path):
    X = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            vals = line.split()
            x = list(map(lambda x: float(x), vals)) 
            X.append(x)
        
        X = np.array(X)
    return X

if __name__ == "__main__":
    train_data = "hw4_nolabel_train.txt"
    train_X = load_data(train_data)

    # Q19
    K = 2
    T = 500
    k_m = K_Means()
    error = 0
    for _ in range(T):
        error += k_m.k_means(train_X, K)

    print("Q19: {}".format(error/T))

    # Q20
    K = 10
    T = 500
    error = 0
    for _ in range(T):
        error += k_m.k_means(train_X, K)

    print("Q20: {}".format(error/T))