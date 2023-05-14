import FixLen
import Fun
### 本文件用于.net文件中信号配时方案的修改
def Green_duration(x):
    f = open('hello2.net.xml', 'r+', encoding='UTF-8')
    flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8, flag9  = 1, 1, 1, 1, 1, 1, 1, 1, 1
    sign1, sign2, sign3, sign4, sign5, sign6, sign7, sign8, sign9 = 0, 3, 6, 9, 12, 15, 18, 21, 24
    intersection = 0
    content = []
    
    for each_line in f:
        ##先判断交叉口
        if '<tlLogic id="J1"' in each_line:
            intersection = 1
        if '<tlLogic id="J2"' in each_line:
            intersection = 2
        if '<tlLogic id="J3"' in each_line:
            intersection = 3
        if '<tlLogic id="J4"' in each_line:
            intersection = 4
        if '<tlLogic id="J5"' in each_line:
            intersection = 5
        if '<tlLogic id="J6"' in each_line:
            intersection = 6
        if '<tlLogic id="J7"' in each_line:
            intersection = 7
        if '<tlLogic id="J8"' in each_line:
            intersection = 8
        if '<tlLogic id="J9"' in each_line:
            intersection = 9           
 
        if intersection == 0:
            content.append(each_line)

        ##交叉口J1
        if intersection == 1:
            x_intersection1 = 120 - 12 - x[0] - x[1] - x[2] 
            if 'phase duration' in each_line:
                if flag1 == 1 or flag1 == 3 or flag1 == 5 or flag1 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag1 == 7:
                        each_line = content1 + '"' + str(x_intersection1) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign1]) + '"' + content3
                    content.append(each_line)
                    flag1 += 1
                    sign1 += 1
                    continue
                if flag1 == 2 or flag1 == 4 or flag1 == 6 or flag1 == 8:
                    flag1 += 1
            content.append(each_line)
            if flag1 == 9:
                intersection = 0
                continue

        ##交叉口J2
        if intersection == 2:
            x_intersection2 = 120 - 12 - x[3] - x[4] - x[5]
            if 'phase duration' in each_line:
                if flag2 == 1 or flag2 == 3 or flag2 == 5 or flag2 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag2 == 7:
                        each_line = content1 + '"' + str(x_intersection2) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign2]) + '"' + content3
                    content.append(each_line)
                    flag2 += 1
                    sign2 += 1
                    continue
                if flag2 == 2 or flag2 == 4 or flag2 == 6 or flag2 == 8:
                    flag2 += 1
            content.append(each_line)
            if flag2 == 9:
                intersection = 0
                continue

        ##交叉口J3
        if intersection == 3:
            x_intersection3 = 120 - 12 - x[6] - x[7] - x[8]
            if 'phase duration' in each_line:
                if flag3 == 1 or flag3 == 3 or flag3 == 5 or flag3 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag3 == 7:
                        each_line = content1 + '"' + str(x_intersection3) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign3]) + '"' + content3
                    content.append(each_line)
                    flag3 += 1
                    sign3 += 1
                    continue
                if flag3 == 2 or flag3 == 4 or flag3 == 6 or flag3 == 8:
                    flag3 += 1
            content.append(each_line)
            if flag3 == 9:
                intersection = 0
                continue

        ##交叉口J4
        if intersection == 4:
            x_intersection4 = 120 - 12 - x[9] - x[10] - x[11]
            if 'phase duration' in each_line:
                if flag4 == 1 or flag4 == 3 or flag4 == 5 or flag4 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag4 == 7:
                        each_line = content1 + '"' + str(x_intersection4) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign4]) + '"' + content3
                    content.append(each_line)
                    flag4 += 1
                    sign4 += 1
                    continue
                if flag4 == 2 or flag4 == 4 or flag4 == 6 or flag4 == 8:
                    flag4 += 1
            content.append(each_line)
            if flag4 == 9:
                intersection = 0
                continue
            
        ##交叉口J5
        if intersection == 5:
            x_intersection5 = 120 - 12 - x[12] - x[13] - x[14]
            if 'phase duration' in each_line:
                if flag5 == 1 or flag5 == 3 or flag5 == 5 or flag5 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag5 == 7:
                        each_line = content1 + '"' + str(x_intersection5) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign5]) + '"' + content3
                    content.append(each_line)
                    flag5 += 1
                    sign5 += 1
                    continue
                if flag5 == 2 or flag5 == 4 or flag5 == 6 or flag5 == 8:
                    flag5 += 1
            content.append(each_line)
            if flag5 == 9:
                intersection = 0
                continue

        ##交叉口J6
        if intersection == 6:
            x_intersection6 = 120 - 12 - x[15] - x[16] - x[17]
            if 'phase duration' in each_line:
                if flag6 == 1 or flag6 == 3 or flag6 == 5 or flag6 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag6 == 7:
                        each_line = content1 + '"' + str(x_intersection6) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign6]) + '"' + content3
                    content.append(each_line)
                    flag6 += 1
                    sign6 += 1
                    continue
                if flag6 == 2 or flag6 == 4 or flag6 == 6 or flag6 == 8:
                    flag6 += 1
            content.append(each_line)
            if flag6 == 9:
                intersection = 0
                continue

        ##交叉口J7
        if intersection == 7:
            x_intersection7 = 120 - 12 - x[18] - x[19] - x[20]
            if 'phase duration' in each_line:
                if flag7 == 1 or flag7 == 3 or flag7 == 5 or flag7 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag7 == 7:
                        each_line = content1 + '"' + str(x_intersection7) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign7]) + '"' + content3
                    content.append(each_line)
                    flag7 += 1
                    sign7 += 1
                    continue
                if flag7 == 2 or flag7 == 4 or flag7 == 6 or flag7 == 8:
                    flag7 += 1
            content.append(each_line)
            if flag7 == 9:
                intersection = 0
                continue

        ##交叉口J8
        if intersection == 8:
            x_intersection8 = 120 - 12 - x[21] - x[22] - x[23]
            if 'phase duration' in each_line:
                if flag8 == 1 or flag8 == 3 or flag8 == 5 or flag8 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag8 == 7:
                        each_line = content1 + '"' + str(x_intersection8) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign8]) + '"' + content3
                    content.append(each_line)
                    flag8 += 1
                    sign8 += 1
                    continue
                if flag8 == 2 or flag8 == 4 or flag8 == 6 or flag8 == 8:
                    flag8 += 1
            content.append(each_line)
            if flag8 == 9:
                intersection = 0
                continue

        ##交叉口J9
        if intersection == 9:
            x_intersection9 = 120 - 12 - x[24] - x[25] - x[26]
            if 'phase duration' in each_line:
                if flag9 == 1 or flag9 == 3 or flag9 == 5 or flag9 == 7:
                    content1, content2, content3 = each_line.split('"', 2)
                    if flag9 == 7:
                        each_line = content1 + '"' + str(x_intersection9) + '"' + content3
                    else:
                        each_line = content1 + '"' + str(x[sign9]) + '"' + content3
                    content.append(each_line)
                    flag9 += 1
                    sign9 += 1
                    continue
                if flag9 == 2 or flag9 == 4 or flag9 == 6 or flag9 == 8:
                    flag9 += 1
            content.append(each_line)
            if flag9 == 9:
                intersection = 0
                continue
               
    f.close()
    file_name = 'hello2.net.xml'
    f_new = open(file_name, 'w', encoding='UTF-8')
    for each in content:
        f_new.writelines(each)
    f_new.close()

if __name__ == "__main__":
    #print("请输入参数x：")
    x = [30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30, 30, 24, 30]
    Green_duration(x)















