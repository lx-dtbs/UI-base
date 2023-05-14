#coding=utf-8
import sys
import os
import re
import GridBase
import DbBase
from   os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)))
import tkinter
import tkinter.filedialog
import subprocess
from   tkinter import *
import Fun
from TRSTI555.TRSTI555.bayes_opt import BayesianOptimization
from TRSTI555.TRSTI555.losser5 import run_R_TuRBO
from PIL import Image
ElementBGArray={}  
ElementBGArray_Resize={} 
ElementBGArray_IM={}

def Button_3_onCommand(uiName,widgetName):
    openPath = tkinter.filedialog.askopenfilename(initialdir=os.path.abspath('.'), title='选择文件',
                                                  filetypes=[('网络数据文件', '*.net.xml'),
                                                             ('All files', '*')])
    if openPath != None and len(openPath) > 0:
        content = Fun.ReadFromFile(openPath)
        Fun.G_CurrentFilePath = openPath
    Fun.SetText(uiName, "Entry_2", openPath)

def Button_4_onCommand(uiName,widgetName):
    openPath = tkinter.filedialog.askopenfilename(initialdir=os.path.abspath('.'), title='选择文件',
                                                  filetypes=[('车流数据文件', '*.rou.xml'),
                                                             ('All files', '*')])
    if openPath != None and len(openPath) > 0:
        content = Fun.ReadFromFile(openPath)
        Fun.G_CurrentFilePath = openPath
    Fun.SetText(uiName, "Entry_4", openPath)

def Button_6_onCommand(uiName,widgetName):
    if os.path.exists("TRSTI555/TRSTI555/Transform_sigal_timing.py"):
      subprocess.Popen(["notepad.exe","TRSTI555/TRSTI555/Transform_sigal_timing.py"])

def Button_10_onCommand(uiName,widgetName):
    openPath = tkinter.filedialog.askopenfilename(initialdir=os.path.abspath('.'), title='选择文件',
                                                  filetypes=[('网络配置文件', '*.sumocfg'),
                                                             ('All files', '*')])
    if openPath != None and len(openPath) > 0:
        content = Fun.ReadFromFile(openPath)
        Fun.G_CurrentFilePath = openPath
    Fun.SetText(uiName, "Entry_11", openPath)
    with open("path/iteration_information.txt", 'w') as f:
        f.truncate(0)

#窗口与SUMO并行
import threading
class ButtonThread(threading.Thread):
    def __init__(self, function):
        threading.Thread.__init__(self)
        self.function = function
    def run(self):
        self.function()


def Button_14_onCommand(uiName,widgetName):
    t=ButtonThread(fixopti)
    global treeview
    treeview = GridBase.clearData(uiName, 'ListView_25')
    t.start()

def fixopti():
    global flag
    flag = 0
    filename="path/iteration_information.txt"
    def TRSTI_Algorithm(x):
        with open(filename,"a+") as f:
            f.write("外部超参数下一迭代点为\n")
            f.write(str(x)+"\n")
        global flag
        flag = flag + 1


        MinWT, x_best = run_R_TuRBO(x, flag)
        with open(filename,"a+") as f:
            f.write("第{0}次迭代目标函数值为：{1}\n".format(flag, MinWT))
            f.write("其所对应的信号配时方案为：{}\n".format(x_best))
        DbBase.deleteAll()
        DbBase.addAccountInfo(flag,MinWT,str(x_best))
        res = DbBase.getData()
        for item in res:
            treeview.insert('', 'end', values=(item[0], item[1], item[2]))
        return -MinWT

    # Bounded region of parameter space
    pbounds = {"x": (0.1, 10)}

    optimizer = BayesianOptimization(
        f=TRSTI_Algorithm,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(
        init_points=3,
        n_iter=30,  #The default maximum number of iterations is 30
        acq='ucb',
        xi=10
    )

    ### While the list of all parameters probed and their corresponding target values is available via the property bo.res.
    for i, res in enumerate(optimizer.res):
        with open(filename, "a+") as f:
            f.write("Iteration {}: \n\t{}\n".format(i, res))

    ### The best combination of parameters and target value found can be accessed via the property bo.max.
    with open(filename,"a+") as f:
        f.write("最好的结果为\n")
        f.write(optimizer.max)

def find_max_numbered_file(dir_path):
    """
    在指定目录中查找以 "result.png" 结尾的文件名中的最大数字，并返回找到的文件名。
    :param dir_path: 要搜索的目录路径
    :return: 找到的文件名，如果没有找到则返回 None
    """
    max_number = None
    max_filename = None
    pattern = r"\d+"

    # 遍历目录中的所有文件名
    for filename in os.listdir(dir_path):
        if filename.endswith("result.jpg"):
            # 获取文件名中的数字部分，并将其转换为整数
            match = re.search(pattern, filename)
            if match is not None:
                number = int(match.group(0))
                if max_number is None or number > max_number:
                    max_number = number
                    max_filename = filename

    return max_filename
def Label_26_onButton1(event,uiName,widgetName):
    dir_path = "pic"
    if os.path.exists("pic/"+find_max_numbered_file(dir_path)):
        Fun.SetImage(uiName,'Label_26',"pic/"+find_max_numbered_file(dir_path))


def Label_26_onDoubleButton1(event,uiName,widgetName):
    op2 = ButtonThread(open2)
    op2.start()
def open2():
    dir_path = "pic"
    with Image.open("pic/"+find_max_numbered_file(dir_path)) as im:
        im.show()

def Button_15_onCommand(uiName,widgetName):
    import traci
    traci.close()