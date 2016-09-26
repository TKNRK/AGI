# 2次元の AGI

import numpy as np
import sympy as sp
from scipy import optimize as opt
from sympy.utilities.lambdify import lambdify
from sympy import Matrix, MatrixSymbol, refine, Identity, Q
from tkinter import *
import time

#  initialize（画像処理関係）
_width = _height = 700  # window's width and height
width = height = 500  # canvas's width and height

# initialize(データ処理関係)
# load adjacency and multi-dimensional space
EdgeList = np.genfromtxt('csv/edgeList.csv', delimiter=",").astype(np.int64)
edge_num = len(EdgeList)
HighDimSpace = np.genfromtxt('csv/mdSpace.csv', delimiter=",")
node_num, high_dim = HighDimSpace.shape

low_dim = 2  # この次元のAGIを実行する

# generate projection vectors
def genE():
    L = np.diag(np.sqrt(np.genfromtxt('csv/eigVals.csv', delimiter=",")[0:high_dim]))
    base = np.zeros(high_dim * low_dim).reshape(high_dim, low_dim)
    for i in range(high_dim): base[i][i % low_dim] = 1
    e0_column = np.zeros(high_dim).reshape(high_dim, 1)
    print(e0_column.shape)
    print(base.shape)
    print(L.shape)
    E = np.c_[L.dot(base), e0_column]
    return E  # 縦ベクトル

Es = genE()  # 射影ベクトルを縦ベクトルで格納(low_dim行が射影ベクトルで、もう１行がベクトル)

Pos_origin = np.array([])  # 計算するデータの実際の座標
Pos_scaled = np.zeros(node_num*low_dim).reshape(low_dim,node_num)  # 画面サイズに合わせたデータの座標
boundingV = 0  # Vertical boundary
boundingH = 0  # Horizontal boundary

def scale(pnt,bool):
	if(bool): return width * (pnt + boundingH / 2) / boundingH + (_width - width) / 2
	else: return (height - 100) * (boundingV / 2 - pnt) / boundingV + (_height - height) / 2

def unscale(pnt,bool):
	if (bool): return boundingH * ((pnt - (_width - width) / 2) - width / 2) / width
	else: return boundingV * ((pnt - (_height - height) / 2) - (height - 100) / 2) / (100 - height)

def update_points_cond(id):
    hds_norm = np.linalg.norm(HighDimSpace[id])
    lds_norm = np.linalg.norm(Pos_origin[id,:])
    if (hds_norm > lds_norm): update_points()

def update_points():
    global Pos_origin, boundingH, boundingV
    Pos_origin = HighDimSpace.dot(Es[:, 0:low_dim])
    boundingH = max([np.amax(Pos_origin[:,0]), abs(np.amin(Pos_origin[:,0]))]) * 2
    boundingV = max([np.amax(Pos_origin[:,1]), abs(np.amin(Pos_origin[:,1]))]) * 2
    for i in range(node_num):
        Pos_scaled[0,i] = scale(Pos_origin[i,0], True);Pos_scaled[1,i] = scale(Pos_origin[i,1], False)

update_points()

print("init: ready")

# sympy
a1,b1,c1,a2,b2,c2,t,s = sp.symbols('a1 b1 c1 a2 b2 c2 t s')   # variables
x_pre,y_pre,x_new,y_new,p_norm = sp.symbols('x_pre y_pre x_new y_new p_norm')  # values
var = (x_pre,y_pre,x_new,y_new,p_norm,a1,b1,c1,a2,b2,c2,t,s)

f = Matrix([
	a1*a1 + b1*b1 + c1*c1*p_norm + 2*(a1*c1*x_pre + b1*c1*y_pre) - 1,
    a2*a2 + b2*b2 + c2*c2*p_norm + 2*(a2*c2*x_pre + b2*c2*y_pre) - 1,
    a1*a2 + b1*b2 + c1*c2*p_norm + x_pre*(a1*c2 + c1*a2) + y_pre*(b1*c2 + c1*b2),
	s*s + t*t - 1,
    s*(a1 + c1*x_pre - 1) + t*(b1 + c1*y_pre),
    s*(a2 + c2*x_pre) + t*(b2 + c2*y_pre - 1),
    a1*x_pre + b1*y_pre + c1*p_norm - x_new,
    a2*x_pre + b2*y_pre + c2*p_norm - y_new
])

func = sp.Matrix.norm(f)
lam_f = lambdify(var, func, 'numpy')

def lam(x_pre,y_pre,x_new,y_new,p_norm):
    return lambda a1,b1,c1,a2,b2,c2,t,s: \
        lam_f(x_pre,y_pre,x_new,y_new,p_norm,a1,b1,c1,a2,b2,c2,t,s)

arr_init = np.array([1, 0, 0, 0, 1, 0, 1, 1])
print("lambda: ready")

######## Graph Drawing ########
root = Tk()
w = Canvas(root, width=_width, height=_height, bg='White')
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
    x2 = unscale(event.x,True)
    y2 = unscale(event.y,False)
    thisID = event.widget.find_withtag(CURRENT)[0] - (edge_num+1)
    p_p = HighDimSpace[thisID].dot(HighDimSpace[thisID])
    Es[:,2] = HighDimSpace[thisID]
    f2 = lam(Pos_origin[thisID,0],Pos_origin[thisID,1],x2, y2, p_p)
    def g(args): return f2(*args)
    res = opt.minimize(g, arr_init, method='L-BFGS-B')
    print(res)
    if (res.success):
        Coefficient = res.x[0:6].reshape(2, 3)
        print(Coefficient)
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