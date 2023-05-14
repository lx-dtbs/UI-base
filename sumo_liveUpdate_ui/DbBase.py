import sqlite3
DbName = "./db/timing_plan.db"
class Cdb:
    def __init__(self,dbName):
        self.dbName = dbName    #数据库名称
        self.conn = None        #文件与数据库的连接
        self.cursor = None      #文件与数据库的交互
        self.__connect()
    def __connect(self):    #将数据库与文件连接
        try:
            self.conn = sqlite3.connect(self.dbName)
            self.cursor = self.conn.cursor()
        except:
            print("conn db err!")
    def exec_query(self,sql,*parms):    #执行命令并查询每行
        try:
            self.cursor.execute(sql,parms)
            values = self.cursor.fetchall()
        except:
            print("exec_query error,sql is=",sql)
            return None
        return values
    def exec_cmd(self,sql,*parms):      #执行命令并保持更改
        try:
            self.cursor.execute(sql,parms)
            self.conn.commit()
        except:
            print("exec_cmd error,sql is=", sql)
    def close_connect(self):            #断开连接
        try:
            self.cursor.close()
            self.conn.close()
        except:
            print("close db err!")
def initDb():   #初始化数据库
    cdb = Cdb(DbName)
    sql = "create table if not exists user (freq integer not null,func integer not null, tp varchar(200))"
    cdb.exec_cmd(sql)
    cdb.close_connect()
def deleteAccount(freq):  #删除指定freq数据
    cdb = Cdb(DbName)
    sql1 = "delete from user where freq=?"
    cdb.exec_cmd(sql1,freq)
    cdb.close_connect()
def deleteAll():
    cdb=Cdb(DbName)
    sql1="delete from user"
    cdb.exec_cmd(sql1)
    cdb.close_connect()
def addAccountInfo(freq,func,tp):  #添加账号信息
    cdb = Cdb(DbName)
    sql1 = "insert into user (freq,func,tp) values (?,?,?)"
    cdb.exec_cmd(sql1, freq, func,tp)
    sql2 = "select max(freq) from user"
    res = cdb.exec_query(sql2)
    cdb.close_connect()
    return res[0][0]
def editAccountInfo(freq,func,tp):  #编辑账号信息
    cdb = Cdb(DbName)
    sql1 = "update user set func=?,tp =? where freq=?"
    cdb.exec_cmd(sql1, freq, func, tp)
    cdb.close_connect()
def getData():  #获取账号信息
    cdb = Cdb(DbName)
    sql2 = "select * from user"
    res = cdb.exec_query(sql2)
    cdb.close_connect()
    return res

if __name__ == '__main__':
    # conn=sqlite3.connect('./db/test.db')
    # c = conn.cursor()  # 获取游标
    # sql = "create table if not exists opti (freq integer not null,func integer not null, timeplan varchar(200))"
    # c.execute(sql)  # 执行sql语句
    # conn.commit()  # 提交数据库操作
    # conn.close()
    # print("ok")

    cdb = Cdb(DbName)
    initDb()
    addAccountInfo(1,80,"[20 30]")
    deleteAll()
    print(getData())
