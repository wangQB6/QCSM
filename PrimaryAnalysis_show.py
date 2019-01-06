# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:02:06 2018

@author: wiser
"""

import pickle
import csv
from sklearn.metrics import roc_curve,auc,roc_auc_score,accuracy_score

import pandas as pd
from pprint import pprint
import pystan
from scipy.special import expit
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签

from numpy.random import normal, randint, binomial, choice, gamma
from numpy import percentile, concatenate, array, linspace, append
from hashlib import md5
import math
import json
import numpy as np


def get_from_params(string,params):
    post = params.get(string)
    post = [[row[i] for row in post] for i in range(len(post[0]))]
    emptyList = []
    for i in range(len(post)):
        emptyList.append(np.mean(post[i]))
    return emptyList
    
def get_csv(path):
    with open(path) as f:
        lines = list(csv.reader(f))
        return lines[0] , lines[1:]

if __name__ == "__main__":
    with open('model.pkl','rb') as f:        
        model = pickle.load(f)
    with open('fit.pkl','rb') as f:        
        fit = pickle.load(f)
    with open('input_data.pkl','rb') as f:        
        input_data = pickle.load(f)
    with open('number_student.pkl','rb') as f:        
        number_student = pickle.load(f)
    student_number = {v: k for k, v in number_student.items()}
    
    
    Z       = input_data['Z']       #总数
    author  = input_data['author']  #做题人学号
    content = input_data['content'] #题号
    judge   = input_data['judge']   #审阅人学号
    grade   = input_data['grade']   #审阅评分
    
    
    params  = fit.extract()
    
    score       = get_from_params('score',params)
    difficult   = get_from_params('difficult',params)
    ability     = get_from_params('ability',params)
    carefulness = get_from_params('carefulness',params)
    bias        = get_from_params('bias',params)
    
    
    
    s_ability = sorted(ability)
    plt.hist(s_ability, 20)
    plt.xlabel('ability')
    plt.ylabel('人数')
    plt.show()
    plt.figure()

    s_bias = sorted(bias)
    plt.hist(s_bias,20)
    plt.xlabel('bias')
    plt.ylabel('人数')
    plt.show()
    plt.figure()


    s_care = sorted(carefulness)
#    plt.plot(s_care,'o')
#    plt.figure()
    plt.hist(s_care,20)
    plt.xlabel('carefulness')
    plt.ylabel('人数')
    plt.show()
    plt.figure()


    s_diff = sorted(difficult)
    plt.hist(s_diff, 20)
    plt.xlabel('difficult')
    plt.ylabel('题目数')
    plt.show()
    plt.figure()

    title,csv_data = get_csv('data/true.csv')
#    ['评为完全正确', '评为存在缺陷', '内部评判', '所提问题', '材料内容', '材料id', '用时', '提问者学号', '提问者姓名']
    
#    对每个人的作答，进行的内部评判，可以看做真实值
    checkList = {}
#    2      5       8
#    答案  id     学号
    for i in csv_data:
        checkList[(int(i[5]),i[7])] = int(i[2])        
    
#    获取每个评审数据的真实值与预测值 
    true = []
    preScore = []
    for i in range(0,Z):
        true.append(checkList[(content[i],number_student[author[i]])])
        temp = 0 if (score[i]<0.49) else 1
        preScore.append(temp)
    
    AUC_1 = roc_auc_score(true,score)
    AUC_2 = roc_auc_score(true,preScore)
    print('AUC is:',AUC_1)
    
    ACC_1 = accuracy_score(true,preScore)
    #ACC_2 = accuracy_score(grade,preScore)
    print('ACC is:',ACC_1)
    



    true_student_score = [0]*61     
    for i in csv_data:
        temp  = i[2]#得分
        true_student_score[student_number[i[7]]-1] += int(i[2])

    student_all_p = []
    for i in range(0,61):
        student_all_p.append([true_student_score[i],ability[i],carefulness[i],number_student[i+1]])
    #转置
    student_all_p = [[row[i] for row in student_all_p] for i in range(len(student_all_p[0]))]
    plt.plot(student_all_p[0],student_all_p[1],'o')
    plt.xlabel('答对题数(共8题)')
    plt.ylabel('ability')
    plt.show()
    
    plt.figure()
    plt.plot(student_all_p[0],student_all_p[2],'o')
    plt.xlabel('答对题数(共8题)')
    plt.ylabel('carefulness')
    plt.show()
    plt.figure()
    



    problem = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0],
               [0,0,0],[0,0,0],[0,0,0],[0,0,0]]     # 答对 总数 百分比 difficult
    for i in csv_data:
        temp = int(i[5])-1 #题号
        problem[temp][1] +=1
        problem[temp][0] += int( i[2])
         
    for i in problem:
        i[2]= i[0]/i[1]
    for i in range(0,len(problem)):
        problem[i].append(difficult[i])
    problem = [[row[i] for row in problem] for i in range(len(problem[0]))]
    plt.plot(problem[2],problem[3],'o')
    plt.xlabel('答对百分比')
    plt.ylabel('difficult')
    plt.show()
    plt.figure()
    

    f = fit.plot('difficult')
    f.set_size_inches(10, 5)
    plt.show()















    
    
    