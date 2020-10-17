def subset(nums, target, sub=[], all_comb=[]):
    # 計算數字組合是否大於等於 36
    # 若等於 36 將組合加到 all_comb 中，
    # 若大於 36 則不用再遞迴下去，結束函式
    sub_sum = sum(sub)
    if sub_sum >= target:
        if sub_sum == target:
            all_comb.append(sub)
        return
    
    # 將新的元素加到子串列中，並遞迴呼叫函式
    remain_size = len(nums)
    for i in range(remain_size):
        new_ele = nums[i]
        remain = nums[i:]
        subset(remain, target, sub+[new_ele], all_comb)
    
    return all_comb
    
nums = [i+1 for i in range(36)]
all_comb = subset(nums, 36)

counts = []
for comb in all_comb:
    all_layer = [10] + comb + [1]
    num_layer = len(all_layer)
    count = 0
    for i in range(num_layer-1):
        count += all_layer[i] * all_layer[i+1]
    counts.append(count)

# Q2
print("Q2 minimum possible number of weights is: {}".format(min(counts)))

# Q3
print("Q3 maximum possible number of weights is: {}".format(max(counts)))