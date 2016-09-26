# 2次元の AGI

import numpy as np
import sympy as sp
from scipy import optimize as opt
from sympy.utilities.lambdify import lambdify, lambdastr
from sympy import Matrix, MatrixSymbol, refine, Identity, Q
from tkinter import *
import time

#  initialize（画像処理関係）
WIDTH = HEIGHT = 700  # window's width and height
width = height = 500  # canvas's width and height

# Calculation Space (C) : 射影更新時の計算における実際の座標系
# Drawing Space (D) : tkinter の描画における画面の座標系

# transfer 'Calculation Space' to 'Drawing Space'
def c2d(pnt, bool):  # データの座標を射影する平面の画面サイズに合わせる
	if(bool): return width * (pnt + boundingH / 2) / boundingH + (WIDTH - width) / 2
	else: return (height - 100) * (boundingV / 2 - pnt) / boundingV + (HEIGHT - height) / 2
# transfer 'Drawing Space' to 'Calculation Space'
def d2c(pnt, bool):  # 射影された平面上の座標を元のスケールに戻す
	if (bool): return boundingH * ((pnt - (WIDTH - width) / 2) - width / 2) / width
	else: return boundingV * ((pnt - (HEIGHT - height) / 2) - (height - 100) / 2) / (100 - height)

# initialize(データ処理関係)
# load adjacency and multi-dimensional space
EdgeList = np.genfromtxt('csv/edgeList.csv', delimiter=",").astype(np.int64)
edge_num = len(EdgeList)
HighDimSpace = np.genfromtxt('csv/mdSpace.csv', delimiter=",")
node_num, high_dim = HighDimSpace.shape

low_dim = 2  # この次元のAGIを実行する

# generate projection vectors
def genE():
    L = np.sqrt(np.genfromtxt('csv/eigVals.csv', delimiter=",")[0:high_dim])
    base = np.zeros(high_dim * low_dim).reshape(low_dim, high_dim)
    e0_column = np.zeros(high_dim).reshape(1,high_dim)
    for i in range(high_dim): base[i % low_dim][i] = 1
    E = np.r_[base*L, e0_column]
    print(e0_column.shape)
    return E.T  # 縦ベクトル

Es = genE()  # 射影ベクトルを縦ベクトルで格納(low_dim行が射影ベクトルで、もう１行がベクトル)

Pos_origin = np.zeros(node_num*low_dim).reshape(node_num,low_dim)  # 計算するデータの実際の座標
Pos_scaled = np.zeros(node_num*low_dim).reshape(low_dim,node_num)  # 画面サイズに合わせたデータの座標
boundingV = 0  # Vertical boundary
boundingH = 0  # Horizontal boundary

def update_points():
    global Pos_origin, boundingH, boundingV
    Pos_origin = HighDimSpace.dot(Es[:, 0:low_dim])
    boundingH = max([np.amax(Pos_origin[:,0]), abs(np.amin(Pos_origin[:,0]))]) * 2
    boundingV = max([np.amax(Pos_origin[:,1]), abs(np.amin(Pos_origin[:,1]))]) * 2
    for i in range(node_num):
        Pos_scaled[0,i] = c2d(Pos_origin[i, 0], True);Pos_scaled[1, i] = c2d(Pos_origin[i, 1], False)

update_points()

print("init: ready")

lam_f = lambda x_pre,y_pre,x_new,y_new,f0_norm,a1,b1,c1,a2,b2,c2,s,t: \
    ((s**2 + t**2 - 1)**2 + (a1*a2 + b1*b2 + c1*c2)**2 + (a1*s + b1*t - s)**2 + (a2*s + b2*t - t)**2 +
     (a1**2 + b1**2 + c1**2 - 1)**2 + (a2**2 + b2**2 + c2**2 - 1)**2 + (a1*x_pre + b1*y_pre + c1*f0_norm - x_new)**2 +
     (a2*x_pre + b2*y_pre + c2*f0_norm - y_new)**2)

def lam(x_pre,y_pre,x_new,y_new,f0_norm):
    return lambda a1,b1,c1,a2,b2,c2,s,t: \
        lam_f(x_pre,y_pre,x_new,y_new,f0_norm,a1,b1,c1,a2,b2,c2,s,t)

arr_init = np.array([1, 0, 0, 0, 1, 0, 1, 1])
print("lambda: ready")

######## Graph Drawing ########
root = Tk()
w = Canvas(root, width=WIDTH, height=HEIGHT, bg='White')
w.pack()
circles = []
lines = []
r = 10

# 初期描画
for e in EdgeList:
	lines.append(w.create_line(Pos_scaled[0,e[0]-1], Pos_scaled[1,e[0]-1],Pos_scaled[0,e[1]-1], Pos_scaled[1,e[1]-1], fill='Black', tags='edge'))
for i in range(node_num):
	circles.append(w.create_oval(Pos_scaled[0,i] - r, Pos_scaled[1,i] - r, Pos_scaled[0,i] + r, Pos_scaled[1,i] + r, fill="White", tags='node'))

# 移動
def move_node(event):
    global Es
    x2 = d2c(event.x, True)
    y2 = d2c(event.y, False)
    thisID = event.widget.find_withtag(CURRENT)[0] - (edge_num+1)
    f0 = HighDimSpace[thisID] - (Pos_origin[thisID, 0] * Es[:, 0] + Pos_origin[thisID, 1] * Es[:, 1])
    f0_norm = np.linalg.norm(f0)
    Es[:, 2] = f0 / f0_norm
    f2 = lam( x2, y2, Pos_origin[thisID,0], Pos_origin[thisID,1], f0_norm)
    def g(args): return f2(*args)
    res = opt.minimize(g, arr_init, method='L-BFGS-B')
    print(res)
    if (res.success):
        Coefficient = res.x[0:6].reshape(2, 3)
        Es[:, 0:2] = Es.dot(Coefficient.T)
        update_points()
        for i in range(node_num):
            w.coords(circles[i], Pos_scaled[0, i] - r, Pos_scaled[1, i] - r, Pos_scaled[0, i] + r, Pos_scaled[1, i] + r)
        for i in range(edge_num):
            w.coords(lines[i], Pos_scaled[0, EdgeList[i][0] - 1], Pos_scaled[1, EdgeList[i][0] - 1],
                     Pos_scaled[0, EdgeList[i][1] - 1], Pos_scaled[1, EdgeList[i][1] - 1])


# バインディング
w.tag_bind('node', '<Button1-Motion>', move_node)

root.mainloop()