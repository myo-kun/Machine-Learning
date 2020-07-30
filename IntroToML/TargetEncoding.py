# No Cross Validation
from sklearn.model_selection import KFold

for c in cat_cols:
    data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
    target_mean = data_tmp.groupby(c)["target"].mean()
    test_x[c] = test_x[c].map(target_mean)

    tmp = np.repeat(np.nan, train_x.shape[0])

    kf = KFold(n_splits=4, shuffle=True, random_state=72)

    for idx_1, idx_2 in kf.split(train_x):
        # out-of-foldで各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    train_x[c] = tmp

# With Cross validation

for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    for c in cat_cols:
        data_tmp = pd.DataFrame({c: tr_x[c], "target": tr_y})
        target_mean = data_tmp.groupby(c)["target"].mean()
        va_x.iloc[:, c] = va_x[c].map(target_mean)

        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()
            tmp[idx_2] = tr_x.iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp