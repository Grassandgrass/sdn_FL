import numpy as np

con_mat=\
    [[   42     ,0    ,21   ,638    ,11     ,0   ,120     ,0     ,0     ,0],
     [    1     ,7    ,34   ,641    ,17     ,0     ,3    ,37     ,0     ,0],
     [    5     ,7   ,347  ,4261    ,52    ,20    ,50    ,63    ,15     ,0],
     [   23     ,5   ,294 ,12087   ,229    ,73   ,186   ,265    ,16     ,0],
     [    3     ,0    ,59  ,1019  ,3724    ,15  ,2198   ,141     ,9     ,0],
     [    1     ,0    ,42   ,316    ,29 ,17384     ,6     ,6     ,3     ,0],
     [   14     ,0    ,39   ,343  ,1458     ,7 ,19534   ,170     ,5     ,0],
     [    0     ,0    ,32   ,923    ,58     ,1    ,89  ,3080     ,3     ,0],
     [    0     ,0     ,9   ,136    ,46     ,2    ,27   ,129   ,118     ,0],
     [    0     ,0     ,2    ,55     ,2     ,0     ,1     ,0     ,0     ,2]]
tpr_list = []
fpr_list = []
con_mat=np.array(con_mat)
for i in range(10):
    number = np.sum(con_mat[:, :])
    tp = con_mat[i][i]
    fn = np.sum(con_mat[i, :]) - tp
    fp = np.sum(con_mat[:, i]) - tp
    tn = number - tp - fn - fp
    #cacc1 = (tp + tn) / number
    tpr1 = tp/(tp + fn)
    fpr1 = fp/(tn + fn)
    # acc_list.append(acc1)
    tpr_list.append(tpr1)
    fpr_list.append(fpr1)
print("tpr:", tpr_list) # 检出率
print("fpr:", fpr_list) # 虚报率