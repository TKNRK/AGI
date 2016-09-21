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
eV = np.array([[0,1]])
eH = np.array([[1,0]])

def scale(pnt,bool):  # データの座標を射影する平面の画面サイズに合わせる
	if(bool): return width * (pnt + boundingH / 2) / boundingH + (_width - width) / 2
	else: return (height - 100) * (boundingV / 2 - pnt) / boundingV + (_height - height) / 2
def unscale(pnt,bool):  # 射影された平面上の座標を元のスケールに戻す
	if (bool): return boundingH * ((pnt - (_width - width) / 2) - width / 2) / width
	else: return boundingV * ((pnt - (_height - height) / 2) - (height - 100) / 2) / (100 - height)

# initialize(データ処理関係)
# load adjacency and multi-dimensional space
EdgeList = np.genfromtxt('csv/edgeList.csv', delimiter=",").astype(np.int64)
edge_num = len(EdgeList)
MDS = np.genfromtxt('csv/mdSpace.csv', delimiter=",")
node_num, high_dim = MDS.shape

low_dim = 2  # この次元のAGIを実行する

# generate projection vectors
def genE():
    L = np.sqrt(np.genfromtxt('csv/eigVals.csv', delimiter=",")[0:high_dim])
    base = np.zeros(high_dim * low_dim).reshape(low_dim, high_dim)
    e0_column = np.zeros(high_dim).reshape(1,high_dim)
    for i in range(high_dim): base[i % low_dim][i] = 1
    E = np.r_[base*L, e0_column]
    return E.T  # 縦ベクトル

Es = genE()  # 射影ベクトルを縦ベクトルで格納(low_dim行が射影ベクトルで、もう１行がベクトル)

Pos_origin = np.zeros(node_num*low_dim).reshape(node_num,low_dim)  # 計算するデータの実際の座標
Pos_scaled = np.zeros(node_num*low_dim).reshape(low_dim,node_num)  # 画面サイズに合わせたデータの座標
boundingV = 0  # Vertical boundary
boundingH = 0  # Horizontal boundary

def scale(pnt,bool):
	if(bool): return width * (pnt + boundingH / 2) / boundingH + (_width - width) / 2
	else: return (height - 100) * (boundingV / 2 - pnt) / boundingV + (_height - height) / 2

def unscale(pnt,bool):
	if (bool): return boundingH * ((pnt - (_width - width) / 2) - width / 2) / width
	else: return boundingV * ((pnt - (_height - height) / 2) - (height - 100) / 2) / (100 - height)

def update_points():
    global Pos_origin, boundingH, boundingV
    Pos_origin = MDS.dot(Es[:,0:low_dim])
    boundingH = max([np.amax(Pos_origin[:,0]), abs(np.amin(Pos_origin[:,0]))]) * 2
    boundingV = max([np.amax(Pos_origin[:,1]), abs(np.amin(Pos_origin[:,1]))]) * 2
    for i in range(node_num):
        Pos_scaled[0,i] = scale(Pos_origin[i,0], True);Pos_scaled[1,i] = scale(Pos_origin[i,1], False)

update_points()

print("init: ready")

# sympy
q =         MatrixSymbol('q', 1, low_dim)  # updated position
p_i =       MatrixSymbol('p_i', 1, high_dim)  # selected point (high_dim) (P[thisID])
E =         MatrixSymbol('E', high_dim,high_dim) # = (e1 e2 e0 0 ...) : 縦ベクトルの列
A_inputEs = MatrixSymbol('A_inputEs', low_dim + 1, low_dim)  # ei' の係数行列
A_inputRs = MatrixSymbol('A_inputRs', low_dim, low_dim - 1)  # Ri の係数行列
var = (q, p_i, E, A_inputEs, A_inputRs)  # 変数のリスト
A = Matrix(np.zeros(high_dim*high_dim).reshape(high_dim,high_dim))
A[0:low_dim + 1, 0:low_dim] = Matrix(A_inputEs)
A[0:low_dim, low_dim:low_dim + 1] = Matrix(A_inputRs)

_E = E * A  # = (e1' e2' R1 0 ...) : 更新後の射影ベクトルは縦ベクトルで格納

# 制約解消における前計算
constraints1 = (refine(_E.T * _E, Q.orthogonal(E))).doit()
constraints2 = (refine(E.T * _E, Q.orthogonal(E))).doit()
constraints3 = p_i * _E

# ei' * ej' = δij (クロネッカーのデルタ)
bases_e = Matrix(constraints1[0:low_dim, 0:low_dim] - Matrix(Identity(low_dim)))
_bases_e = bases_e[0,0]**2 + bases_e[0,1]**2 + bases_e[1,1]**2

# Ri * Rj = δij (クロネッカーのデルタ)
bases_r = Matrix(constraints1[low_dim:2 * low_dim - 1, low_dim:2 * low_dim - 1] - Matrix(Identity(low_dim - 1)))
_bases_r = bases_r[0,0]**2

# _Ei.dot(Rj) - sp.Matrix(Ei).dot(Rj),
eMulR =  Matrix(constraints1[0:low_dim, low_dim:2 * low_dim - 1] - constraints2[0:low_dim, low_dim:2 * low_dim - 1])
_eMulR = eMulR[0,0]**2 + eMulR[1,0]**2

# sp.Matrix(P_i).dot(_Ej) - wj
pew = Matrix(constraints3[0, 0:low_dim] - q)
_pew = pew[0,0]**2 + pew[0,1]**2

func = _bases_e + _bases_r + _eMulR + _pew  # 制約式の二乗和
lam_f = lambdify(var, func, 'numpy')

def lam(q_, P_i, Esub):
    """
    q_    :(1,low_dim) ドラッグされた行先の点の座標
    Esub  :(high_dim,low_dim+1) 画面が更新される前の基底とe0の行列
    P_i   :(1,high_dim) ドラッグされた点に対応する高次元座標
    """
    # a[:,0:3]
    E_ = np.zeros(high_dim*high_dim).reshape(high_dim,high_dim)  # <- これをなくしたい！
    E_[0:high_dim, 0:low_dim + 1] = Esub
    return lambda A_1, A_2: lam_f(q_, P_i, E_, A_1, A_2)

arr_init = np.array([1,0,0,1,0,0,0,0])  # どう一般化するか？
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
    if(low_dim == 3): return 0
    position = x2*eH + y2*eV
    thisID = event.widget.find_withtag(CURRENT)[0] - (edge_num+1)
    #E_0 = MDS[thisID] - (x2 * Es[:,0] + y2 * Es[:,1])
    E_0 = MDS[thisID] - np.sum(position.dot(Es[:, 0:low_dim].T), axis=1)
    Es[:,low_dim] = E_0 / np.linalg.norm(E_0)
    f2 = lam(position, MDS[thisID].reshape(1, high_dim), Es)
    def g(args):
        """
        最適化関数
        :param args: 行列を行ベクトルに崩したもの
        :return: f2 に args を叩き込んだもの
        """
        arr1 = args[0:low_dim*(low_dim+1)].reshape(low_dim+1, low_dim)  # E' variables ( E'[0] = this[0:dim+1,0] * (E:E0) )
        arr2 = args[low_dim*(low_dim+1):2*low_dim**2].reshape(low_dim, low_dim-1)  # R  variables ( ignore )
        return f2(arr1, arr2)
    res = opt.minimize(g, arr_init, method='L-BFGS-B')  # ,options={'ftol':1e-3}
    if(res.success):
        Coefficient = res.x[0:(low_dim+1)*low_dim].reshape(low_dim+1,low_dim)
        Es[:,0:low_dim] = Es.dot(Coefficient)
        update_points()
        for i in range(node_num):
            w.coords(circles[i], Pos_scaled[0,i] - r, Pos_scaled[1,i] - r, Pos_scaled[0,i] + r, Pos_scaled[1,i] + r)
        for i in range(edge_num):
            w.coords(lines[i], Pos_scaled[0,EdgeList[i][0] - 1], Pos_scaled[1,EdgeList[i][0] - 1], Pos_scaled[0,EdgeList[i][1] - 1], Pos_scaled[1,EdgeList[i][1] - 1])

# バインディング
w.tag_bind('node', '<Button1-Motion>', move_node)
root.mainloop()
