import sys
import numpy as np
from RegularizedKMeans import RegularizedKMeans
import funs as Ifuns
import funs_metric as Mfuns

X, y_true, N, dim, c_true = Ifuns.load_mat("./dataset/wine.mat")
print(N, dim, c_true)


model = RegularizedKMeans(X=X.astype(np.float64), c_true=c_true, init_method=b"random_y", warm_start=True, n_jobs=6, seed=0)
model.opt(rep=10, type=b"Hard")
print(np.mean(model.obj))

acc = Mfuns.multi_accuracy(y_true, Y=model.y_pre)
print(np.mean(acc))

# paper: 2.962e+6 (obj)
# run  : 2.962e+6 (obj)
