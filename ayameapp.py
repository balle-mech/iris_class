import tkinter as tk
from tkinter import messagebox
from sklearn import datasets
from sklearn import svm

#ウィンドウの設定
root = tk.Tk()
root.title("アヤメの品種分類")
root.geometry("300x300")

lbl = tk.Label(text="がくの長さ(Sepal Length)")
lbl.place(x=20, y= 70)

lbl = tk.Label(text="がくの幅 (Sepal Width)")
lbl.place(x=20, y= 100)

lbl = tk.Label(text="花びらの長さ (Petal Length)")
lbl.place(x=20, y= 130)

lbl = tk.Label(text="花びらの幅 (Petal Width)")
lbl.place(x=20, y= 160)

#単位
for num in range(4):
    lbl = tk.Label(text="cm")
    lbl.place(x=250, y= 70+30*num)

#アヤメの特徴量取得
txt = tk.Entry(width=10)
txt.place(x=180, y=70)

txt1 = tk.Entry(width=10)
txt1.place(x=180, y=100)

txt2 = tk.Entry(width=10)
txt2.place(x=180, y=130)

txt3 = tk.Entry(width=10)
txt3.place(x=180, y=160)

# ボタンクリック時に呼び出す関数を定義
def ayame_class():
    a = txt.get()
    b = txt1.get()
    c = txt2.get()
    d = txt3.get()
    
    #Irisの測定データの読み込み
    iris = datasets.load_iris()
    clf = svm.LinearSVC()
    clf.fit(iris.data, iris.target) #第1引数に特徴量、第2引数にラベルデータ
    classfy = clf.predict([[float(a), float(b), float(c), float(d)]])
    variety = ["Setosa（セトサ）", "Versicolor（バージカラー）",  "Versinica（バージニカ）"]
    
    result = tk.Label(text = "品種は" + variety[int(classfy)] + "です!")
    result.place(x=60, y=240)

# ボタンの作成,commandにて実行関数を指定
button = tk.Button(text="品種分類スタート", command=ayame_class)
# placeプロパティにて配置箇所をX,Y座標で指定
button.place(x=100, y=200)

root.mainloop()
