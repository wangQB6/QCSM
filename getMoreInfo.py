# -*- coding: utf-8 -*-

#

import pickle
import csv
import math
from matplotlib import pyplot as plt

def save(theOne,fileName):
    with open(fileName,'wb') as f:
        pickle.dump(theOne,f)

def get_csv(path):
    with open(path) as f:
        lines = list(csv.reader(f))
        return lines[0] , lines[1:]

def get_material_score(path):
    """
    return a dict with 
    material id  as key  and 
    [trueNum, falseNum, difficultPoint] as value
    in which difficultPoint = falseNum /( trueNum + falseNum )
    """
    
    titie , data =  get_csv(path)
    
    ### 统计每道题的打分情况
    temp = -1#暂存为材料id最大值
    for i in data:
        material_id = int(i[5])
        temp = [temp,material_id][temp<material_id]
        
    all_material_score = {}
    #譬如 all_material_score[1] = [3,9]  代表对于id为1的材料，所有“提问”中，3人次认为是对，9人次认为存在缺陷
    for i in range(1,temp+1):#初始化
        all_material_score[i] = [0,0]
    
    for i in data:
        material_id = int(i[5])
        if(i[0]=='-1' or i[1]=='-1'):
            pass
        else:
            all_material_score[material_id][0] += int(i[0])
            all_material_score[material_id][1] += int(i[1])
    
    for i in all_material_score:
        temp = all_material_score[i]
        all_material_score[i].append(temp[1]/(temp[0]+temp[1]))
    return all_material_score,  [all_material_score[i][2] for i in range(1,len(all_material_score)+1)]

def get_student_ability_simple(path):
    titie , data =  get_csv(path)
    
    student_ability = {}
    #譬如 student_ability['123'] = [3,9]  代表对于id为123的学生，所有评价中，3人次认为是对，9人次认为存在缺陷
    for i in data:
        if(i[0]=='-1' or i[1]=='-1'):
            continue
        s_id = i[7]#type is string
        if(s_id in student_ability):
            student_ability[s_id][0] += int(i[0])
            student_ability[s_id][1] += int(i[1])
        else:
            student_ability[s_id] = [int(i[0]),int(i[1])]
    
    for i in student_ability:
        temp = student_ability[i]
        student_ability[i].append(temp[0]/(temp[0]+temp[1]))
    return student_ability

def get_student_ability(path):
    temp , material_score = get_material_score(path)
    titie , data =  get_csv(path)
    student_ability = {}
    #譬如 student_ability['123'] = [3,9]  代表对于id为123的学生，所有评价中，3人次认为是对，9人次认为存在缺陷
    for i in data:
        if(i[0]=='-1' or i[1]=='-1'):
            continue
        s_id = i[7]#type is string
        if(s_id in student_ability):
            student_ability[s_id][0] += int(i[0]) * material_score[int(i[5])-1]
            student_ability[s_id][1] += int(i[1]) * material_score[int(i[5])-1]
        else:
            student_ability[s_id] = [  int(i[0])* material_score[int(i[5])-1],
                                       int(i[1])* material_score[int(i[5])-1]]
    for i in student_ability:
        temp = student_ability[i]
        student_ability[i].append(temp[0]/(temp[0]+temp[1]))
    return student_ability
 
if __name__ == "__main__":
    for i in range(1,7):
        path = 'data/judgement'+str(i)+'.csv'
        temp, material_score = get_material_score(path)
        ability_simple  = get_student_ability_simple(path) 
        ability         = get_student_ability(path)
        plt.hist(material_score)
        plt.show()
        plt.figure()
        print('mean difficult point',i,'is',sum(material_score)/len(material_score))
















