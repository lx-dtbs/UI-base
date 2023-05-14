import Fun
# TreeView作为表格使用时的相关操作
def addItem(uiName,TreeViewName,*item):
    treeview = Fun.GetElement(uiName,TreeViewName)
    treeview.insert('', 'end', values=(item[0], item[1], item[2]))
def editSelected(uiName,TreeViewName,*item):
    treeview = Fun.GetElement(uiName,TreeViewName)
    index = treeview.selection()
    if(len(index) == 0):
        return None
    treeview.set(index, '用户账号', item[0])
    treeview.set(index, '密码', item[1])
    return treeview.item(index)['values']
def deleteSelected(uiName,TreeViewName):
    treeview = Fun.GetElement(uiName,TreeViewName)
    index = treeview.selection()
    if(len(index) == 0):
        return None
    item = treeview.item(index)
    treeview.delete(index)
    return item['values']
def getSelected(uiName,TreeViewName):
    treeview = Fun.GetElement(uiName,TreeViewName)
    index = treeview.selection()
    if(len(index) == 0):
        return None
    return treeview.item(index)['values']
def clearData(uiName,TreeViewName):
    treeview = Fun.GetElement(uiName,TreeViewName)
    obj = treeview.get_children()
    for i in obj:
        treeview.delete(i)
    return treeview
'''
def loopData(uiName,TreeViewName):
    treeview = Fun.GetElement(uiName,TreeViewName)
    obj = treeview.get_children()
    for i in obj:
        item = treeview.item(i)
        print(item['values'])
'''
