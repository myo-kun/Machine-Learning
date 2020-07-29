# Fair function and Pseudo-Huber function
# XGBではMAEが目的関数としてつかえない為、そのための近似

def fair(preds, dtrain):
    x = preds - dtrain.get_lables()
    c = 1.0 # parameter
    den = abs(x) + c
    grad = c * x / den
    hess = c * c / den ** 2
    return grad, hess

def pseudo_huber(preds, dtrain):
    d = preds - dtrain.get_lables()
    delta = 1.0 # parameter
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


