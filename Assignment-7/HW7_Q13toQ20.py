import time
import numpy as np

class Node:
    """
    Node for decision tree.

    param:
    hypo: hypothesis of branch if it's a root of a sub-tree
    value: return value if it's a leaf node
    left: left sub-tree
    right: right sub-tree
    """
    def __init__(self, hypo, value=None, left=None, right=None):
        self.hypo = hypo
        self.value = value
        self.left = left
        self.right = right
    
class DecisionTree:
    """
    C&RT applied DT with Gini Index impurity measure.
    """ 
    def __init__(self):
        self.leave = 0  # 紀錄有多少個葉節點
        self.internal = 0 # 紀錄有多少內部節點
        
    def cart(self, X, Y, prune=False):
        # 若 Y 全部相等，或只剩一個資料點，或 X 全部相同則停止遞迴
        if np.unique(Y).shape[0] == 1 or X.shape[1] == 1 or np.all(X==X[:,0][:,np.newaxis]):
            self.leave += 1
            
            # 計算 +1, -1 何者較多 (bincount 不接受負值，所以要先+1再扣除)
            y = np.bincount(Y+1).argmax() - 1
            node = Node(hypo=None, value=y)
            return node

        # 產生樹高等於 1 的決策樹    
        if prune == True:
            # 計算 best hypothesis，並依照 hypothesis 將資料切成兩份
            theta, dim = self.decision_stump(X, Y)
            left_pos, right_pos = X[dim, :] <= theta, X[dim, :] > theta
            left_X, right_X = X[:, left_pos], X[:, right_pos]
            left_Y, right_Y = Y[left_pos], Y[right_pos]

            # 建立新的子樹
            node = Node(hypo=(theta, dim))
            left_value = 1 if np.sum(left_Y) > 0 else -1
            right_value = 1 if np.sum(right_Y) > 0 else -1

            left = Node(hypo=None, value=left_value)
            right = Node(hypo=None, value=right_value)
            node.left = left
            node.right = right
            return node

        # 計算 best hypothesis，並依照 hypothesis 將資料切成兩份
        theta, dim = self.decision_stump(X, Y)
        left_pos, right_pos = X[dim, :] <= theta, X[dim, :] > theta
        left_X, right_X = X[:, left_pos], X[:, right_pos]
        left_Y, right_Y = Y[left_pos], Y[right_pos]

        # 建立新的子樹
        self.internal += 1
        node = Node(hypo=(theta, dim))
        node.left = self.cart(left_X, left_Y)
        node.right = self.cart(right_X, right_Y)
        
        return node

    def decision_stump(self, X, Y):
        cat = [-1, 1]
        dim = X.shape[1]

        best_b = float('inf')
        best_theta = 0
        best_dim = 0
        for i in range(X.shape[0]):
            sort_i = np.argsort(X[i])
            sort_X = X[i][sort_i]
            sort_Y = Y[sort_i]

            theta_n = np.array([float('-inf')] + [(sort_X[i]+sort_X[i+1])/2 for i in range(dim-1)] + [float('inf')])
            for theta in theta_n:
                left_Y = sort_Y[sort_X < theta]
                right_Y = sort_Y[sort_X > theta]

                b = left_Y.shape[0]*self.gini(left_Y, cat)+right_Y.shape[0]*self.gini(right_Y, cat)
                if b < best_b:
                    best_b = b
                    best_theta = theta
                    best_dim = i
                
        return best_theta, best_dim

    def gini(self, Y, catogories):
        data_size = Y.shape[0]
        if data_size == 0:
            return 0
        purity = 0
        for c in catogories:
            purity += ((sum(Y==c))/(data_size))**2
        return 1-purity

    def buildTree(self, train_X, train_Y, prune=False):
        root = self.cart(train_X, train_Y, prune)
        return root

    def valid(self, test_X, test_Y, root):
        error = 0
        values = []
        for data in zip(*test_X, test_Y):
            # 從 root 節點出發進行判斷
            node = root
            X, y = data[:-1], data[-1]

            # 當非葉節點時，依條件決定往左或右子樹走
            while node.value is None:
                theta, dim = node.hypo
                if X[dim] < theta:
                    node = node.left
                else:
                    node = node.right

            # 達到葉節點後，進行判斷  
            values.append(node.value)
            if y != node.value:
                error += 1

        return error/test_Y.shape[0], values

class RandomForest:
    def __init__(self):
        self.DT = DecisionTree()

    def bagging(self, train_X, train_Y, N):
        # 取得向量維數
        dim = train_Y.shape[0]
        # 隨機產生 N 個 index(< dim)
        np.random.seed(int(time.time()))
        rand_index = np.random.randint(0, dim, size=N)
        # 依 index bagging
        bag_X = train_X[:, rand_index]
        bag_Y = train_Y[rand_index]

        return bag_X, bag_Y

    def forest(self, train_X, train_Y, N, prune=False):
        dim = train_Y.shape[0]
        trees = []
        for _ in range(N):
            bag_X, bag_Y = self.bagging(train_X, train_Y, dim)
            tree = self.DT.buildTree(bag_X, bag_Y, prune)
            trees.append(tree)
        return trees

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
        
        X = np.array(X).T
        Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    train_file = "hw3_dectree_train.txt"
    test_file = "hw3_dectree_test.txt"
    train_X, train_Y = load_data(train_file)
    test_X, test_Y = load_data(test_file)

    DT = DecisionTree()
    root = DT.buildTree(train_X, train_Y)
    # Q13
    print("Q13. 內部節點數量為: ", DT.internal)
    # Q14
    Ein, _ = DT.valid(train_X, train_Y, root)
    print("Q14. Ein為: ", Ein)
    # Q15
    Eout, _ = DT.valid(test_X, test_Y, root)
    print("Q15. Eout為: ", Eout)
    #------------------------------------------#
    # 題目要求 100 次太花時間，僅運行 3 次
    F = 3
    T = 300
    Ein_g = []
    Ein_RF = []
    Eout_RF = []
    RF = RandomForest()
    for _ in range(F):
        # 初始化 RF 計算 Ein &　Ｅout 的串列
        RF_in = []
        RF_out = []
        # 生成 T 棵樹組成的隨機森林
        trees = RF.forest(train_X, train_Y, T)
        for tree in trees:
            # 計算 error 以及對應產生的輸出
            ein, val_in = RF.DT.valid(train_X, train_Y, tree)
            eout, val_out = RF.DT.valid(test_X, test_Y, tree)
            # 將輸出整併至串列
            Ein_g.append(ein)
            RF_in.append(val_in)
            RF_out.append(val_out)

        # 將隨機森林產生的輸出，轉成以 -1 & 1 組成的陣列
        RF_in, RF_out = np.array(RF_in), np.array(RF_out)
        G_in, G_out = np.sum(RF_in, axis=0), np.sum(RF_out, axis=0)
        G_in, G_out = np.sign(G_in), np.sign(G_out)
        
        # 計算隨機森林的 Ein 及 Eout
        ein_rf = sum(G_in!=train_Y)/train_Y.shape[0]
        eout_rf = sum(G_out!=test_Y)/test_Y.shape[0]
        Ein_RF.append(ein_rf)
        Eout_RF.append(eout_rf)
    
    # Q16
    print("Q16. Ein(gt)的平均錯誤率為: ", sum(Ein_g)/len(Ein_g))
    # Q17
    print("Q17. Ein(RF)的平均錯誤率為: ", sum(Ein_RF)/len(Ein_RF))
    # Q18
    print("Q18. Eout(RF)的平均錯誤率為: ", sum(Eout_RF)/len(Eout_RF))
    #------------------------------------------#
    Ein_RF = []
    Eout_RF = []
    RF = RandomForest()
    for _ in range(F):
        # 初始化 RF 計算 Ein &　Ｅout 的串列
        RF_in = []
        RF_out = []
        # 生成 T 棵樹組成的隨機森林
        trees = RF.forest(train_X, train_Y, T, prune=True)
        for tree in trees:
            # 計算 error 以及對應產生的輸出
            ein, val_in = RF.DT.valid(train_X, train_Y, tree)
            eout, val_out = RF.DT.valid(test_X, test_Y, tree)
            # 將輸出整併至串列
            RF_in.append(val_in)
            RF_out.append(val_out)

        # 將隨機森林產生的輸出，轉成以 -1 & 1 組成的陣列
        RF_in, RF_out = np.array(RF_in), np.array(RF_out)
        G_in, G_out = np.sum(RF_in, axis=0), np.sum(RF_out, axis=0)
        G_in, G_out = np.sign(G_in), np.sign(G_out)
        
        # 計算隨機森林的 Ein 及 Eout
        ein_rf = sum(G_in!=train_Y)/train_Y.shape[0]
        eout_rf = sum(G_out!=test_Y)/test_Y.shape[0]
        Ein_RF.append(ein_rf)
        Eout_RF.append(eout_rf)

    # Q19
    print("Q19. Ein(RF)的平均錯誤率為: ", sum(Ein_RF)/len(Ein_RF))
    # Q20
    print("Q20. Eout(RF)的平均錯誤率為: ", sum(Eout_RF)/len(Eout_RF))