#coding=utf-8
#import libs 
import sys
import FixLen_cmd
import Fun
import os
import tkinter
from   tkinter import *
import tkinter.ttk
import tkinter.font
import ctypes
#Add your Varial Here: (Keep This Line of comments)
#Define UI Class

class  FixLen:
    def __init__(self,root,isTKroot = True):
        uiName = self.__class__.__name__
        self.uiName = uiName
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        Fun.Register(uiName,'UIClass',self)
        self.root = root
        self.isTKroot = isTKroot
        Fun.Register(uiName,'root',root)
        if isTKroot == True:
            root.title("定长信号优化")
            if os.path.exists("Resources/zxyc.ico"):
                root.iconbitmap("Resources/zxyc.ico")
            Fun.CenterDlg(uiName,root,705,555)
            root['background'] = '#efefef'
            root.bind('<Configure>',self.Configure)
        Form_1= tkinter.Canvas(root,width = 10,height = 4)
        Form_1.pack(fill=BOTH,expand=True)
        Form_1.configure(bg = "#efefef")
        Form_1.configure(highlightthickness = 0)
        Fun.Register(uiName,'Form_1',Form_1)
        #Create the elements of root
        Entry_2_Variable = Fun.AddTKVariable(uiName,'Entry_2','')
        Entry_2 = tkinter.Entry(Form_1,textvariable=Entry_2_Variable)
        Fun.Register(uiName,'Entry_2',Entry_2)
        Fun.SetControlPlace(uiName,'Entry_2',524,185,127,28)
        Entry_2.configure(relief = "sunken")
        Entry_2.configure(state="readonly")
        Button_3 = tkinter.Button(Form_1,text="网络文件")
        Fun.Register(uiName,'Button_3',Button_3)
        Fun.SetControlPlace(uiName,'Button_3',374,185,100,28)
        Button_3.configure(command=lambda: FixLen_cmd.Button_3_onCommand(uiName, "Button_3"))
        Entry_4_Variable = Fun.AddTKVariable(uiName, 'Entry_4', '')
        Entry_4 = tkinter.Entry(Form_1, textvariable=Entry_4_Variable)
        Fun.Register(uiName, 'Entry_4', Entry_4)
        Fun.SetControlPlace(uiName, 'Entry_4', 524, 275, 127, 28)
        Entry_4.configure(relief="sunken")
        Entry_4.configure(state="readonly")
        Button_4 = tkinter.Button(Form_1, text="车流文件")
        Fun.Register(uiName, 'Button_4', Button_4)
        Fun.SetControlPlace(uiName, 'Button_4', 374, 275, 100, 28)
        Button_4.configure(command=lambda: FixLen_cmd.Button_4_onCommand(uiName, "Button_4"))
        ComboBox_8_Variable = Fun.AddTKVariable(uiName,'ComboBox_8')
        ComboBox_8 = tkinter.ttk.Combobox(Form_1,textvariable=ComboBox_8_Variable, state="readonly")
        Fun.Register(uiName,'ComboBox_8',ComboBox_8)
        Fun.SetControlPlace(uiName,'ComboBox_8',173,99,100,20)
        ComboBox_8["values"]=['网格型','其他']
        ComboBox_8.current(0)
        Label_9 = tkinter.Label(Form_1,text="网络类型")
        Fun.Register(uiName,'Label_9',Label_9)
        Fun.SetControlPlace(uiName,'Label_9',31,99,100,20)
        Label_9.configure(relief = "flat")
        Button_10 = tkinter.Button(Form_1,text="网络配置文件")
        Fun.Register(uiName,'Button_10',Button_10)
        Fun.SetControlPlace(uiName,'Button_10',374,95,100,28)
        Button_10.configure(command=lambda: FixLen_cmd.Button_10_onCommand(uiName, "Button_10"))
        Entry_11_Variable = Fun.AddTKVariable(uiName,'Entry_11','')
        Entry_11 = tkinter.Entry(Form_1,textvariable=Entry_11_Variable)
        Fun.Register(uiName,'Entry_11',Entry_11)
        Fun.SetControlPlace(uiName,'Entry_11',524,95,127,28)
        Entry_11.configure(relief = "sunken",state="readonly")
        Label_12 = tkinter.Label(Form_1,text="定长信号算法优化")
        Fun.Register(uiName,'Label_12',Label_12)
        Fun.SetControlPlace(uiName,'Label_12',239,32,230,28)
        Label_12.configure(relief = "flat")
        Label_12_Ft=tkinter.font.Font(family='System', size=15,weight='bold',slant='roman',underline=0,overstrike=0)
        Label_12.configure(font = Label_12_Ft)
        Button_14 = tkinter.Button(Form_1,text="优化")
        Fun.Register(uiName,'Button_14',Button_14)
        Fun.SetControlPlace(uiName,'Button_14',43,387,100,28)
        Button_14.configure(command=lambda: FixLen_cmd.Button_14_onCommand(uiName, "Button_14"))
        Button_15 = tkinter.Button(Form_1, text="停止")
        Fun.Register(uiName, 'Button_15', Button_15)
        Fun.SetControlPlace(uiName, 'Button_15', 43, 457, 100, 28)
        Button_15.configure(command=lambda: FixLen_cmd.Button_15_onCommand(uiName, "Button_15"))
        ListView_25 = tkinter.ttk.Treeview(Form_1,show="headings")
        Fun.Register(uiName,'ListView_25',ListView_25)
        Fun.SetControlPlace(uiName,'ListView_25',192,361,459,160)
        ListView_25.configure(selectmode = "extended")
        ListView_25.configure(columns = ["次数","迭代目标函数值","信号配时方案"])
        ListView_25.column("次数",anchor="center",width=10)
        ListView_25.heading("次数",anchor="center",text="次数")
        ListView_25.column("迭代目标函数值",anchor="center",width=20)
        ListView_25.heading("迭代目标函数值",anchor="center",text="迭代目标函数值")
        ListView_25.column("信号配时方案",anchor="center",width=200)
        ListView_25.heading("信号配时方案",anchor="center",text="信号配时方案")

        import pyperclip
        def copy_to_clipboard(event):
            # 获取点击的列和行
            column_index = ListView_25["columns"].index("信号配时方案")
            row_index = str(ListView_25.identify_row(event.y))

            # 获取点击行所在的记录
            item = ListView_25.item(row_index)

            # 获取指定列的内容并复制到剪贴板
            text_to_copy = item["values"][column_index]
            pyperclip.copy(text_to_copy)

        tags = list(ListView_25.bindtags())
        tags.insert(2, "Copy")
        ListView_25.bindtags(tuple(tags))
        ListView_25.bind_class("Copy", "<Button-1>", copy_to_clipboard)

        Label_26 = tkinter.Label(Form_1)
        Fun.Register(uiName,'Label_26',Label_26)
        Fun.SetControlPlace(uiName,'Label_26',52,145,279,157)
        Label_26.configure(bg = "#ffffff")
        Label_26.configure(relief = "flat")
        Label_26.bind("<Button-1>",Fun.EventFunction_Adaptor(FixLen_cmd.Label_26_onButton1, uiName=uiName, widgetName="Label_26"))
        Label_26.bind("<Double-Button-1>",Fun.EventFunction_Adaptor(FixLen_cmd.Label_26_onDoubleButton1, uiName=uiName, widgetName="Label_26"))
        Label_26.lift()
        Label_27 = tkinter.Label(Form_1,text="配时优化图像")
        Fun.Register(uiName,'Label_27',Label_27)
        Fun.SetControlPlace(uiName,'Label_27',148,309,100,20)
        Label_27.configure(relief = "flat")
        #Inital all element's Data 
        Fun.InitElementData(uiName)
        #Add Some Logic Code Here: (Keep This Line of comments)



        #Exit Application: (Keep This Line of comments)
        if self.isTKroot == True:
            self.root.protocol('WM_DELETE_WINDOW', self.Exit)
    def Exit(self):
        if self.isTKroot == True:
            self.root.destroy()

    def Configure(self,event):
        if self.root == event.widget:
            pass
#Create the root of Kinter 
if  __name__ == '__main__':
    root = tkinter.Tk()
    MyDlg = FixLen(root)
    root.mainloop()