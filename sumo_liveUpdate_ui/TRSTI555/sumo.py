
import os, sys
import numpy as np
import operator
from functools import reduce
import Transform_sigal_timing

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

import optparse

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def myfunction(x,u):
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    ##打开配置文件
    traci.start([sumoBinary, "-c", "hello.sumocfg"])

    #周期时间为120秒，运行时间为4500秒
    '''
    构造的ts_infor四维矩阵说明：
    交叉口数×单个交叉口检测器数量×30个周期（即30个120秒共计3600秒）×一个周期内统计的排队长度车辆的次数
    其中第一个参数为网络中信号控制交叉口的数量；
    第二个参数为每个交叉口的进口道E2检测器的数量，
    该参数编号规则0→北直，1→北右，2→南直，3→南右，4→北左，5→南左，6→西直，7－西右，8－东直，9→东右，10→西左，11→东左；
    第三个参数为需要统计周期的数；
    第四个参数为每个周期内统计排队车辆长度的次数。
    '''
    num_inter = 9
    coll = 120 ### 最大的绿灯时长
    ts_infor = np.zeros((num_inter,12,30,coll))
    Waiting_time = 0
    edgeID = []
    f = open('edgeID.txt')
    for eachline in f:
        edgeID.append(eachline.split('\n')[0])
    f.close

    for step in range(0, 4500):
        simStep = traci.simulation.getTime()
        if simStep >= 900:
            n = int((simStep-900)/120)
            for eachedge in edgeID:
                Waiting_time += traci.edge.getWaitingTime(eachedge)
            
            '''
            因为我们的原则是以指标的最差情况作为统计依据，所以在这里我直接取交叉口周期时段内所有仿真步的车辆排队数量，之后去max
            '''
            index = step - 900 - 120*n
            ### 获取交叉口J1的状态
            ts_infor[0][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E5_1")/23
            ts_infor[0][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E5_0")/23
            ts_infor[0][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E8_1")/23
            ts_infor[0][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E8_0")/23

            ts_infor[0][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E5_2")/23
            ts_infor[0][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E8_2")/23           

            ts_infor[0][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E4_1")/23
            ts_infor[0][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E4_0")/23
            ts_infor[0][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E3_1")/23
            ts_infor[0][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E3_0")/23
                
            ts_infor[0][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E4_2")/23
            ts_infor[0][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E3_2")/23      
            
            ### 获取交叉口J2的状态
            ts_infor[1][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E6_1")/23
            ts_infor[1][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E6_0")/23
            ts_infor[1][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E9_1")/23
            ts_infor[1][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E9_0")/23

            ts_infor[1][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E6_2")/23
            ts_infor[1][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E9_2")/23           

            ts_infor[1][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E3_1")/23
            ts_infor[1][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E3_0")/23
            ts_infor[1][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E1_1")/23
            ts_infor[1][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E1_0")/23
           
            ts_infor[1][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E3_2")/23
            ts_infor[1][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E1_2")/23 

            ### 获取交叉口J3的状态
            ts_infor[2][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E7_1")/23
            ts_infor[2][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E7_0")/23
            ts_infor[2][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E10_1")/23
            ts_infor[2][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E10_0")/23

            ts_infor[2][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E7_2")/23
            ts_infor[2][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E10_2")/23           

            ts_infor[2][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E1_1")/23
            ts_infor[2][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E1_0")/23
            ts_infor[2][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E2_1")/23
            ts_infor[2][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E2_0")/23
              
            ts_infor[2][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E1_2")/23
            ts_infor[2][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E2_2")/23 

            ### 获取交叉口J4的状态
            ts_infor[3][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E8_1")/23
            ts_infor[3][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E8_0")/23
            ts_infor[3][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E15_1")/23
            ts_infor[3][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E15_0")/23

            ts_infor[3][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E8_2")/23
            ts_infor[3][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E15_2")/23           

            ts_infor[3][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E11_1")/23
            ts_infor[3][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E11_0")/23
            ts_infor[3][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E12_1")/23
            ts_infor[3][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E12_0")/23
             
            ts_infor[3][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E11_2")/23
            ts_infor[3][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E12_2")/23     

            ### 获取交叉口J5的状态
            ts_infor[4][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E9_1")/23
            ts_infor[4][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E9_0")/23
            ts_infor[4][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E16_1")/23
            ts_infor[4][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E16_0")/23

            ts_infor[4][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E9_2")/23
            ts_infor[4][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E16_2")/23           

            ts_infor[4][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E12_1")/23
            ts_infor[4][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E12_0")/23
            ts_infor[4][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E13_1")/23
            ts_infor[4][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E13_0")/23
              
            ts_infor[4][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E12_2")/23
            ts_infor[4][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E13_2")/23

            ### 获取交叉口J6的状态
            ts_infor[5][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E10_1")/23
            ts_infor[5][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E10_0")/23
            ts_infor[5][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E17_1")/23
            ts_infor[5][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E17_0")/23

            ts_infor[5][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E10_2")/23
            ts_infor[5][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E17_2")/23           

            ts_infor[5][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E13_1")/23
            ts_infor[5][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E13_0")/23
            ts_infor[5][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E14_1")/23
            ts_infor[5][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E14_0")/23
               
            ts_infor[5][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E13_2")/23
            ts_infor[5][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E14_2")/23  

            ### 获取交叉口J7的状态
            ts_infor[6][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E15_1")/23
            ts_infor[6][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E15_0")/23
            ts_infor[6][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E19_1")/23
            ts_infor[6][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E19_0")/23

            ts_infor[6][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E15_2")/23
            ts_infor[6][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E19_2")/23           

            ts_infor[6][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E18_1")/23
            ts_infor[6][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E18_0")/23
            ts_infor[6][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E20_1")/23
            ts_infor[6][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E20_0")/23
                
            ts_infor[6][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E18_2")/23
            ts_infor[6][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E20_2")/23

            ### 获取交叉口J8的状态
            ts_infor[7][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E16_1")/23
            ts_infor[7][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E16_0")/23
            ts_infor[7][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E21_1")/23
            ts_infor[7][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E21_0")/23

            ts_infor[7][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E16_2")/23
            ts_infor[7][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E21_2")/23           

            ts_infor[7][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E20_1")/23
            ts_infor[7][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E20_0")/23
            ts_infor[7][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E22_1")/23
            ts_infor[7][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E22_0")/23
              
            ts_infor[7][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E20_2")/23
            ts_infor[7][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E22_2")/23  

            ### 获取交叉口J9的状态
            ts_infor[8][0][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E17_1")/23
            ts_infor[8][1][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E17_0")/23
            ts_infor[8][2][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E23_1")/23
            ts_infor[8][3][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E23_0")/23

            ts_infor[8][4][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E17_2")/23
            ts_infor[8][5][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E23_2")/23           

            ts_infor[8][6][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E22_1")/23
            ts_infor[8][7][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E22_0")/23
            ts_infor[8][8][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E24_1")/23
            ts_infor[8][9][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E24_0")/23
             
            ts_infor[8][10][n][index] = traci.lanearea.getJamLengthVehicle("e2det_E22_2")/23
            ts_infor[8][11][n][index] = traci.lanearea.getJamLengthVehicle("e2det_-E24_2")/23                    
        traci.simulationStep()
    ### print(ts_infor)
    '''
    交叉口对周围交叉口的影响水平权重说明：
    行表示交叉口数量，列表示交叉口相位数
    '''
    weight_ts = np.zeros((9,4))
    ### 计算交叉口J1相位对周围交叉口的影响水平
    weight_ts[0][0] = max((np.max(ts_infor[0][2]))*(np.max(ts_infor[3][10])+np.max(ts_infor[3][2])+np.max(ts_infor[3][9]))*0.6/u, np.max(ts_infor[0][2]))
    weight_ts[0][1] = max((np.max(ts_infor[0][5]))*(np.max(ts_infor[3][10])+np.max(ts_infor[3][2])+np.max(ts_infor[3][9]))*0.3/u, np.max(ts_infor[0][5]))
    weight_ts[0][2] = max((np.max(ts_infor[0][8]))*(np.max(ts_infor[1][1])+np.max(ts_infor[1][8])+np.max(ts_infor[1][5]))*0.6/u, np.max(ts_infor[0][8]))
    weight_ts[0][3] = max((np.max(ts_infor[0][11]))*(np.max(ts_infor[1][1])+np.max(ts_infor[1][8])+np.max(ts_infor[1][5]))*0.3/u, np.max(ts_infor[0][11]))
    ### 计算交叉口J2相位对周围交叉口的影响水平
    weight_ts[1][0] = max((np.max(ts_infor[1][2]))*(np.max(ts_infor[4][10])+np.max(ts_infor[4][2])+np.max(ts_infor[4][9]))*0.6/u, np.max(ts_infor[1][2]))
    weight_ts[1][1] = max((np.max(ts_infor[1][5]))*(np.max(ts_infor[4][10])+np.max(ts_infor[4][2])+np.max(ts_infor[4][9]))*0.3/u, np.max(ts_infor[1][5]))
    weight_ts[1][2] = max((np.max(ts_infor[1][6]))*(np.max(ts_infor[0][4])+np.max(ts_infor[0][6])+np.max(ts_infor[0][3]))*0.6/u, np.max(ts_infor[1][6]),(np.max(ts_infor[1][8]))*(np.max(ts_infor[2][1])+np.max(ts_infor[2][8])+np.max(ts_infor[2][5]))*0.6/u, np.max(ts_infor[1][8]))
    weight_ts[1][3] = max((np.max(ts_infor[1][10]))*(np.max(ts_infor[0][4])+np.max(ts_infor[0][6])+np.max(ts_infor[0][3]))*0.3/u, np.max(ts_infor[1][10]),(np.max(ts_infor[1][11]))*(np.max(ts_infor[2][1])+np.max(ts_infor[2][8])+np.max(ts_infor[2][5]))*0.3/u, np.max(ts_infor[1][11]))
    ### 计算交叉口J3相位对周围交叉口的影响水平
    weight_ts[2][0] = max((np.max(ts_infor[2][2]))*(np.max(ts_infor[5][10])+np.max(ts_infor[5][2])+np.max(ts_infor[5][9]))*0.6/u, np.max(ts_infor[2][2]))
    weight_ts[2][1] = max((np.max(ts_infor[2][5]))*(np.max(ts_infor[5][10])+np.max(ts_infor[5][2])+np.max(ts_infor[5][9]))*0.3/u, np.max(ts_infor[2][5]))
    weight_ts[2][2] = max((np.max(ts_infor[2][6]))*(np.max(ts_infor[1][4])+np.max(ts_infor[1][6])+np.max(ts_infor[1][3]))*0.6/u, np.max(ts_infor[2][6]))
    weight_ts[2][2] = max((np.max(ts_infor[2][10]))*(np.max(ts_infor[1][4])+np.max(ts_infor[1][6])+np.max(ts_infor[1][3]))*0.3/u, np.max(ts_infor[2][10]))
    ### 计算交叉口J4相位对周围交叉口的影响水平
    weight_ts[3][0] = max((np.max(ts_infor[3][0]))*(np.max(ts_infor[0][7])+np.max(ts_infor[0][0])+np.max(ts_infor[0][11]))*0.6/u, np.max(ts_infor[3][0]),(np.max(ts_infor[3][2]))*(np.max(ts_infor[6][10])+np.max(ts_infor[6][2])+np.max(ts_infor[6][9]))*0.6/u, np.max(ts_infor[3][2]))
    weight_ts[3][1] = max((np.max(ts_infor[3][4]))*(np.max(ts_infor[0][7])+np.max(ts_infor[0][0])+np.max(ts_infor[0][11]))*0.3/u, np.max(ts_infor[3][4]),(np.max(ts_infor[3][5]))*(np.max(ts_infor[6][10])+np.max(ts_infor[6][2])+np.max(ts_infor[6][9]))*0.3/u, np.max(ts_infor[3][5]))
    weight_ts[3][2] = max((np.max(ts_infor[3][8]))*(np.max(ts_infor[4][1])+np.max(ts_infor[4][8])+np.max(ts_infor[4][5]))*0.6/u, np.max(ts_infor[3][8]))
    weight_ts[3][2] = max((np.max(ts_infor[3][11]))*(np.max(ts_infor[4][1])+np.max(ts_infor[4][8])+np.max(ts_infor[4][5]))*0.3/u, np.max(ts_infor[3][11]))
    ### 计算交叉口J5相位对周围交叉口的影响水平
    weight_ts[4][0] = max((np.max(ts_infor[4][0]))*(np.max(ts_infor[1][7])+np.max(ts_infor[1][0])+np.max(ts_infor[1][11]))*0.6/u, np.max(ts_infor[4][0]),(np.max(ts_infor[4][2]))*(np.max(ts_infor[7][10])+np.max(ts_infor[7][2])+np.max(ts_infor[7][9]))*0.6/u, np.max(ts_infor[4][2]))
    weight_ts[4][1] = max((np.max(ts_infor[4][4]))*(np.max(ts_infor[1][7])+np.max(ts_infor[1][0])+np.max(ts_infor[1][11]))*0.3/u, np.max(ts_infor[4][4]),(np.max(ts_infor[4][5]))*(np.max(ts_infor[7][10])+np.max(ts_infor[7][2])+np.max(ts_infor[7][9]))*0.3/u, np.max(ts_infor[4][5]))
    weight_ts[4][2] = max((np.max(ts_infor[4][6]))*(np.max(ts_infor[3][4])+np.max(ts_infor[3][6])+np.max(ts_infor[3][3]))*0.6/u, np.max(ts_infor[4][6]),(np.max(ts_infor[4][8]))*(np.max(ts_infor[5][1])+np.max(ts_infor[5][8])+np.max(ts_infor[5][5]))*0.6/u, np.max(ts_infor[4][8]))
    weight_ts[4][3] = max((np.max(ts_infor[4][10]))*(np.max(ts_infor[3][4])+np.max(ts_infor[3][6])+np.max(ts_infor[3][3]))*0.3/u, np.max(ts_infor[4][10]),(np.max(ts_infor[4][11]))*(np.max(ts_infor[5][1])+np.max(ts_infor[5][8])+np.max(ts_infor[5][5]))*0.3/u, np.max(ts_infor[4][11]))
    ### 计算交叉口J6相位对周围交叉口的影响水平
    weight_ts[5][0] = max((np.max(ts_infor[5][0]))*(np.max(ts_infor[2][7])+np.max(ts_infor[2][0])+np.max(ts_infor[2][11]))*0.6/u, np.max(ts_infor[5][0]),(np.max(ts_infor[5][2]))*(np.max(ts_infor[8][10])+np.max(ts_infor[8][2])+np.max(ts_infor[8][9]))*0.6/u, np.max(ts_infor[5][2]))
    weight_ts[5][1] = max((np.max(ts_infor[5][4]))*(np.max(ts_infor[2][7])+np.max(ts_infor[2][0])+np.max(ts_infor[2][11]))*0.3/u, np.max(ts_infor[5][4]),(np.max(ts_infor[5][5]))*(np.max(ts_infor[8][10])+np.max(ts_infor[8][2])+np.max(ts_infor[8][9]))*0.3/u, np.max(ts_infor[5][5]))
    weight_ts[5][2] = max((np.max(ts_infor[5][6]))*(np.max(ts_infor[4][4])+np.max(ts_infor[4][6])+np.max(ts_infor[4][3]))*0.6/u, np.max(ts_infor[5][6]))
    weight_ts[5][3] = max((np.max(ts_infor[5][10]))*(np.max(ts_infor[4][4])+np.max(ts_infor[4][6])+np.max(ts_infor[4][3]))*0.3/u, np.max(ts_infor[5][10]))
    ### 计算交叉口J7相位对周围交叉口的影响水平
    weight_ts[6][0] = max((np.max(ts_infor[6][0]))*(np.max(ts_infor[3][7])+np.max(ts_infor[3][0])+np.max(ts_infor[0][11]))*0.6/u, np.max(ts_infor[6][0]))
    weight_ts[6][1] = max((np.max(ts_infor[6][4]))*(np.max(ts_infor[3][7])+np.max(ts_infor[3][0])+np.max(ts_infor[0][11]))*0.3/u, np.max(ts_infor[6][4]))
    weight_ts[6][2] = max((np.max(ts_infor[6][8]))*(np.max(ts_infor[7][1])+np.max(ts_infor[7][8])+np.max(ts_infor[7][5]))*0.6/u, np.max(ts_infor[6][8]))
    weight_ts[6][3] = max((np.max(ts_infor[6][11]))*(np.max(ts_infor[7][1])+np.max(ts_infor[7][8])+np.max(ts_infor[7][5]))*0.3/u, np.max(ts_infor[6][11]))
    ### 计算交叉口J8相位对周围交叉口的影响水平 
    weight_ts[7][0] = max((np.max(ts_infor[7][0]))*(np.max(ts_infor[4][7])+np.max(ts_infor[4][0])+np.max(ts_infor[4][11]))*0.6/u, np.max(ts_infor[7][0]))
    weight_ts[7][1] = max((np.max(ts_infor[7][4]))*(np.max(ts_infor[4][7])+np.max(ts_infor[4][0])+np.max(ts_infor[4][11]))*0.3/u, np.max(ts_infor[7][4]))
    weight_ts[7][2] = max((np.max(ts_infor[7][6]))*(np.max(ts_infor[6][4])+np.max(ts_infor[6][6])+np.max(ts_infor[6][3]))*0.6/u, np.max(ts_infor[7][6]),(np.max(ts_infor[7][8]))*(np.max(ts_infor[8][1])+np.max(ts_infor[8][8])+np.max(ts_infor[8][5]))*0.6/u, np.max(ts_infor[7][8]))
    weight_ts[7][3] = max((np.max(ts_infor[7][10]))*(np.max(ts_infor[6][4])+np.max(ts_infor[6][6])+np.max(ts_infor[6][3]))*0.3/u, np.max(ts_infor[7][10]),(np.max(ts_infor[7][11]))*(np.max(ts_infor[8][1])+np.max(ts_infor[8][8])+np.max(ts_infor[8][5]))*0.3/u, np.max(ts_infor[7][11]))
    ### 计算交叉口J9相位对周围交叉口的影响水平
    weight_ts[8][0] = max((np.max(ts_infor[8][0]))*(np.max(ts_infor[3][7])+np.max(ts_infor[3][0])+np.max(ts_infor[0][11]))*0.6/u, np.max(ts_infor[8][0]))
    weight_ts[8][1] = max((np.max(ts_infor[8][4]))*(np.max(ts_infor[3][7])+np.max(ts_infor[3][0])+np.max(ts_infor[0][11]))*0.3/u, np.max(ts_infor[8][4]))
    weight_ts[8][2] = max((np.max(ts_infor[8][6]))*(np.max(ts_infor[7][4])+np.max(ts_infor[7][6])+np.max(ts_infor[7][3]))*0.6/u, np.max(ts_infor[8][6]))
    weight_ts[8][3] = max((np.max(ts_infor[8][10]))*(np.max(ts_infor[7][4])+np.max(ts_infor[7][6])+np.max(ts_infor[7][3]))*0.3/u, np.max(ts_infor[8][10]))
    SP = np.empty([9,3])
    for i in range(9):
        for j in range(3):
            SP[i][j] = weight_ts[i][j]/(np.sum(weight_ts)/36)
    weights = SP.reshape(-1)
    traci.close()
    return Waiting_time, weights

if __name__ == "__main__":
    '''
    X1=np.array([17.47085248, 19.28308844, 26.29426345,  9.55849966, 22.47886246, 28.63202314,
                13.4394535,  29.67523383, 32.50548392, 23.62231398, 24.80106315, 34.29372095,
                23.41855851, 24.85578969, 31.59506267,  6.73396141, 28.46556354, 15.2577476,
                23.30098189, 15.06733992, 28.66881894, 12.51466812, 22.19939271, 17.13942663,
                19.40839862, 29.93290426, 25.15701973])
    '''
    X2=np.array([31.03298992, 20.37424972,  7.3255602,  31.44739598, 27.45130197, 6.43673568,
                 30.13522994,  9.3749032,  27.10450662, 12.27800398, 10.57463327, 15.87764075,
                 30.74288428, 14.23228725, 26.32131726, 13.8818539,  18.09372224, 23.64758401,
                 27.46181412, 27.03020977, 17.14838976, 31.50358179, 12.89336774, 20.32220473,
                 21.59196779,  7.9707037,   7.98179566])
    Transform_sigal_timing.Green_duration(X2)
    y1, y2 = myfunction(X2)
    print(y1, y2)
    ''' X1结果
    91102918.0 [0.01020036 0.01020036 0.01202186 0.03205829 0.03205829 0.04031573
 0.02756527 0.02756527 0.04116576 0.01153613 0.01153613 0.03205829
 0.03849423 0.03849423 0.04808743 0.05537341 0.05537341 0.02635094
 0.01153613 0.01153613 0.01408622 0.0472374  0.0472374  0.02659381
 0.01153613 0.01153613 0.05039466]
    '''
    ''' X2结果
    91102918.0 [0.01020036 0.01020036 0.01202186 0.03205829 0.03205829 0.04031573
 0.02756527 0.02756527 0.04116576 0.01153613 0.01153613 0.03205829
 0.03849423 0.03849423 0.04808743 0.05537341 0.05537341 0.02635094
 0.01153613 0.01153613 0.01408622 0.0472374  0.0472374  0.02659381
 0.01153613 0.01153613 0.05039466]
    '''
