from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# データの読み込み
iris = datasets.load_iris()
x, y = iris.data, iris.target

# トレーニングデータとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# 前処理:正規化(各項目の値を0~1に統一)
print("Min: {}".format(x_train.min(axis=0) [:3]) + ", Max: {}".format(x_train.max(axis=0) 
[:3]))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_s = scaler.fit(x_train).transform(x_train) # x_train の正規化
print("Min: {}".format(x_train_s.min(axis=0)[:3])) # 正規化後の最小値
print("Max: {}".format(x_train_s.max(axis=0)[:3])) # 正規化後の最大値
X_test_s = scaler.transform(x_test) # x_test の正規化

# モデルの選択
model = svm.LinearSVC()

# 学習
model.fit(x_train, y_train)

# テスト
pred = model.predict(x_test)

# 評価：正答率,適合率,再現率
from sklearn.metrics import classification_report as CR
print("{}".format(CR(y_test, pred, target_names = iris.target_names)))

# 学習済みモデルを使う
print(model.predict([[1.4, 3.5, 5.1, 0.2], [6.5, 2.6, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))