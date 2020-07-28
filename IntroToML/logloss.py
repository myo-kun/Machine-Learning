# Implementing logloss for XGBoost

import xgboost as xgb
from sklearn.metrics import log_loss

# Transform features and target to xgboost data structure

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(val_x, label=val_y)

# カスタム目的関数
# xgboostの'binary:logistic'と同じ
def logregobj(pred, dtrain):
    labels = dtrain.get_label()
    # シグモイド関数
    pred = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

# カスタム評価指数
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)

params = {'silent': 1, 'random_state': 71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)
pred_val = bst.predict(dvalid)
pred = 1.0 / (1.0 + np.exp(-pred_val))
logloss = log_loss(val_y, pred)
print(logloss)

# Normal way
params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(val_y, pred)
print(logloss)
