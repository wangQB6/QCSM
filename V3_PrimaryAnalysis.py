
# -*- coding: utf-8 -*-
import pickle
import pandas as pd
from pprint import pprint
import pystan
from scipy.special import expit
from matplotlib import pyplot as plt
from numpy.random import normal, randint, binomial, choice, gamma
from numpy import percentile, concatenate, array, linspace, append
from hashlib import md5
import math
import json
import numpy as np

def save(theOne,fileName):
    with open(fileName,'wb') as f:
        pickle.dump(theOne,f)


# 处理为stan输入数据
def process_input():
    # 读取数据
    with open("data/lab.json", 'r') as f:
        data = json.load(f)

    # 学号和编号的映射
    student_number = {}
    number_student = {}
    count = 0
    for key in data.keys():
        count += 1
        student_number[key] = count
        number_student[count] = key

    # 写入数据
    author_data = []
    content_data = []
    judge_data = []
    grade_data = []
    for author in data:
        for content in data[author]:
            if 'review' in data[author][content]:
                for judge in data[author][content]['review']:
                    if judge in student_number.keys():
                        author_data.append(student_number[author])
                        content_data.append(int(content))
                        judge_data.append(student_number[judge])
                        grade_data.append(data[author][content]['review'][judge]['view'])

    #print('------------------')
    #for i in range(len(grade_data)):
    #    print(author_data[i], '-', content_data[i],'<-',judge_data[i],'\t:\t',grade_data[i])

    input_data = {
        'V': len(data),
        'X': 32,  # 32道题目
        'U': len(data),
        'Z': len(grade_data),

        'author': author_data,
        'content': content_data,
        'judge': judge_data,
        'grade': grade_data,

        'a_0': 0.5,
        'na_0': 20,

        'd_0': 0.33,
        'nd_0': 20,

        'c_0': 0.5,
        'nc_0': 100,

        'nb_0': 1000,

        'lambda': 0.001
    }
    return input_data, number_student


if __name__ == "__main__":
    input_data, number_student = process_input()
    save(input_data,'input_data.pkl')
    save(number_student,'number_student.pkl')
#
    # stan code
    model_code = '''
    data {
        // numbers
        int<lower=1> V;
        int<lower=1> X;
        int<lower=1> U;
        int<lower=1> Z;
    
        // data
        int<lower=1, upper=V> author[Z];
        int<lower=1, upper=X> content[Z];
        int<lower=1, upper=U> judge[Z];
        int<lower=0,upper=1> grade[Z];
    
        // hyper-parameters
        real<lower=0> c_0;
        real<lower=0> nc_0;
        real<lower=0> a_0;
        real<lower=0> na_0;
        real<lower=0> d_0;
        real<lower=0> nd_0;
        real<lower=0> nb_0;
        real<lower=0> lambda;
    }
    
    parameters {
        // parameters
        real<lower=0,upper=1> difficult[X];
        real<lower=0,upper=1> ability[V];
        real<lower=0,upper=1> carefulness[V];
        real<lower= -1,upper=1> bias[V];
        real<lower=0,upper=1> score[Z];
    }
    
    model {
		real quality;
        // priors
        difficult ~ normal(d_0,sqrt(1 / nd_0));
        ability ~ normal(a_0,sqrt(1/ na_0));
        carefulness ~ normal(ability,sqrt(1/nc_0));
        bias ~ normal(0,sqrt(1/nb_0));
        
        // data model
        for (i in 1:Z){
			  quality = 1 / (1 + exp ((-1.7) * 1.7 * carefulness[author[i]]* (ability[author[i]] - difficult[content[i]])));
              score[i] ~ normal(quality + bias[judge[i]],sqrt(lambda / (carefulness[judge[i]]))); 
              //lambda/ (ability[judge[i]])  0.2 * ( 0.5 - bias[judge[i]] )
              grade[i] ~ bernoulli(score[i]);
        }
    }
    '''
    model = pystan.StanModel(model_code=model_code)
    save(model,'model.pkl')
    # fit model
    fit = model.sampling(data=input_data, iter=150, chains=3) #500,4
    save(fit,'fit.pkl')
    # extract parameters
    params = fit.extract()

    # print fit summary
    #print(fit)
    # draw summary plot

    #f = fit.plot()
    #f.set_size_inches(18, 10)
    #plt.tight_layout()



    post_difficult = params.get('difficult')
    # 转置
    post_difficult = [[row[i] for row in post_difficult] for i in range(len(post_difficult[0]))]
    predict = []
    for i in range(len(post_difficult)):
        predict.append(np.mean(post_difficult[i]))
    for i in range(len(predict)):
        print('task:', i+1, '\tdiffcult：', predict[i])

    post_ability = params.get('ability')
    post_ability = [[row[i] for row in post_ability] for i in range(len(post_ability[0]))]
    predict = []
    for i in range(len(post_ability)):
        predict.append(np.mean(post_ability[i]))

    for i in range(len(predict)):
        print('student:', number_student[i + 1], '\tability：', predict[i])

  #  post_carefulness = params.get('carefulness')
   # post_carefulness = [[row[i] for row in post_carefulness] for i in range(len(post_carefulness[0]))]
  #  predict = []
  #  for i in range(len(post_carefulness)):
   #     predict.append(np.mean(post_carefulness[i]))

   # for i in range(len(predict)):
#        print('student:', number_student[i + 1], '\tcarefulness：', predict[i])


