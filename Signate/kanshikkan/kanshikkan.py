import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sns.set()


def preprocess_sex(df):
    df["Gender"] = df["Gender"].replace({"Male": 0, "Female": 1})


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

out_id = test_data["id"]
train_data["AG_ratio"] = train_data["AG_ratio"].fillna(train_data["AG_ratio"].mean())
preprocess_sex(train_data)
y_train = train_data["disease"]
X_train = train_data.drop(
    columns=["id", "Age", "disease", "Alb", "AG_ratio", "Gender", "TP"]
)
X_test = test_data.drop(columns=["id", "Age", "Alb", "AG_ratio", "Gender", "TP"])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

output = dict(zip(out_id.to_list(), predictions))
submission = pd.DataFrame.from_dict(output, orient="index")
submission.to_csv("submission.csv", header=False)
