import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator

###########4321ADE############
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)

# plt.title('Parameter Sensitivity Analysis of MFAE on ETH/UCY Datasets \n under the univariate setting (History Length $L$ = 8, Initial $\mathregular{N_1N_2N_3N_4 = 4321}$)',fontsize=23,family='Times New Roman')
# plt.tick_params(axis='both',which='major',labelsize=14)
#Numbers of Self-attention Layers for Input at Different Scales
x1 = [4,5,6,7,8] #x
x2 = [3,4,5,6,7,8] 
x4 = [2,3,4,5,6,7,8] 
x8 = [1,2,3,4,5,6,7,8] 

y1 = [0.50,0.49,0.47,0.49,0.48]
y2 = [0.50,0.48,0.46,0.47,0.48,0.48]
y4 = [0.50,0.48,0.50,0.52,0.53,0.53,0.54]
y8 = [0.50,0.48,0.49,0.50,0.52,0.53,0.51,0.52]

# #传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
# plt.plot(x,y,color='red',alpha=0.3,linestyle='--',linewidth=5,marker='o'
#          ,markeredgecolor='r',markersize='20',markeredgewidth=10)

#'''
plt.plot(x1, y1, 'o--', alpha=0.7, linewidth=2, color='green',\
          markersize=10,label='$\mathregular{N_1}$(L-scale)')
#'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x2, y2, 'D--', alpha=0.7, linewidth=2, color='red',\
          markersize=9,label='$\mathregular{N_2}$(L/2-scale)')
plt.plot(x4, y4, '*--', alpha=0.7, linewidth=2, color='purple',\
          markersize=12,label='$\mathregular{N_3}$(L/4-scale)')
plt.plot(x8, y8, 's--', alpha=0.7, linewidth=2, color='orange',\
          markersize=9,label='$\mathregular{N_4}$(L/8-scale)')


x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0.6,8.4)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0.455,0.545)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderaxespad=0, prop=font1)  #显示上面的label

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}

plt.xlabel('The number of attention layers',font2) #x_label
plt.ylabel('Average $\mathregular{ADE_{20}}$',font2)#y_label
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()


###########4321FDE############
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)

# plt.title('Parameter Sensitivity Analysis of MFAE on ETH/UCY Datasets \n under the univariate setting (History Length $L$ = 8, Initial $\mathregular{N_1N_2N_3N_4 = 4321}$)',fontsize=23,family='Times New Roman')
# plt.tick_params(axis='both',which='major',labelsize=14)

x1 = [4,5,6,7,8] #x
x2 = [3,4,5,6,7,8] 
x4 = [2,3,4,5,6,7,8] 
x8 = [1,2,3,4,5,6,7,8] 

y1 = [1.02,0.99,0.95,0.97,0.98]
y2 = [1.02,0.98,0.94,0.95,0.97,0.99]
y4 = [1.02,0.98,0.99,1.01,1.01,1.03,1.02]
y8 = [1.02,0.99,1.01,1.00,1.00,1.02,1.03,1.05]

# #传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
# plt.plot(x,y,color='red',alpha=0.3,linestyle='--',linewidth=5,marker='o'
#          ,markeredgecolor='r',markersize='20',markeredgewidth=10)

#'''
plt.plot(x1, y1, 'o--', alpha=0.7, linewidth=2, color='green',\
          markersize=10,label='$\mathregular{N_1}$(L-scale)')
#'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x2, y2, 'D--', alpha=0.7, linewidth=2, color='red',\
          markersize=9,label='$\mathregular{N_2}$(L/2-scale)')
plt.plot(x4, y4, '*--', alpha=0.7, linewidth=2, color='purple',\
          markersize=12,label='$\mathregular{N_3}$(L/4-scale)')
plt.plot(x8, y8, 's--', alpha=0.7, linewidth=2, color='orange',\
          markersize=9,label='$\mathregular{N_4}$(L/8-scale)')


x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0.6,8.4)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0.935,1.055)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderaxespad=0, prop=font1)  #显示上面的label

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}

plt.xlabel('The number of attention layers',font2) #x_label
plt.ylabel('Average $\mathregular{FDE_{20}}$',font2)#y_label
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

##########6532ADE############
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)

# plt.title('Parameter Sensitivity Analysis of MFAE on ETH/UCY Datasets \n under the univariate setting (History Length $L$ = 8, Initial $\mathregular{N_1N_2N_3N_4 = 6532}$)',fontsize=23,family='Times New Roman')
# plt.tick_params(axis='both',which='major',labelsize=14)

x1 = [4,5,6,7,8] #x
x2 = [3,4,5,6,7,8] 
x4 = [2,3,4,5,6,7,8] 
x8 = [1,2,3,4,5,6,7,8] 

y1 = [0.44,0.42,0.41,0.42,0.44]
y2 = [0.44,0.43,0.41,0.43,0.45,0.44]
y4 = [0.43,0.41,0.42,0.43,0.43,0.44,0.46]
y8 = [0.44,0.41,0.43,0.42,0.44,0.43,0.44,0.45]

# #传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
# plt.plot(x,y,color='red',alpha=0.3,linestyle='--',linewidth=5,marker='o'
#          ,markeredgecolor='r',markersize='20',markeredgewidth=10)

#'''
plt.plot(x1, y1, 'o--', alpha=0.7, linewidth=2, color='green',\
          markersize=10,label='$\mathregular{N_1}$(L-scale)')
#'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x2, y2, 'D--', alpha=0.7, linewidth=2, color='red',\
          markersize=9,label='$\mathregular{N_2}$(L/2-scale)')
plt.plot(x4, y4, '*--', alpha=0.7, linewidth=2, color='purple',\
          markersize=12,label='$\mathregular{N_3}$(L/4-scale)')
plt.plot(x8, y8, 's--', alpha=0.7, linewidth=2, color='orange',\
          markersize=9,label='$\mathregular{N_4}$(L/8-scale)')


x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0.6,8.4)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0.405,0.465)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderaxespad=0, prop=font1)  #显示上面的label

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}

plt.xlabel('The number of attention layers',font2) #x_label
plt.ylabel('Average $\mathregular{ADE_{20}}$',font2)#y_label
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

##########6532FDE############
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)

# plt.title('Parameter Sensitivity Analysis of MFAE on ETH/UCY Datasets \n under the univariate setting (History Length $L$ = 8, Initial $\mathregular{N_1N_2N_3N_4 = 6532}$)',fontsize=23,family='Times New Roman')
# plt.tick_params(axis='both',which='major',labelsize=14)

x1 = [4,5,6,7,8] #x
x2 = [3,4,5,6,7,8] 
x4 = [2,3,4,5,6,7,8] 
x8 = [1,2,3,4,5,6,7,8] 

y1 = [0.91,0.88,0.86,0.88,0.90]
y2 = [0.91,0.89,0.86,0.88,0.89,0.91]
y4 = [0.89,0.86,0.88,0.89,0.90,0.91,0.90]
y8 = [0.88,0.86,0.87,0.88,0.89,0.90,0.90,0.92]

# #传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
# plt.plot(x,y,color='red',alpha=0.3,linestyle='--',linewidth=5,marker='o'
#          ,markeredgecolor='r',markersize='20',markeredgewidth=10)

#'''
plt.plot(x1, y1, 'o--', alpha=0.7, linewidth=2, color='green',\
          markersize=10,label='$\mathregular{N_1}$(L-scale)')
#'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x2, y2, 'D--', alpha=0.7, linewidth=2, color='red',\
          markersize=9,label='$\mathregular{N_2}$(L/2-scale)')
plt.plot(x4, y4, '*--', alpha=0.7, linewidth=2, color='purple',\
          markersize=12,label='$\mathregular{N_3}$(L/4-scale)')
plt.plot(x8, y8, 's--', alpha=0.7, linewidth=2, color='orange',\
          markersize=9,label='$\mathregular{N_4}$(L/8-scale)')


x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0.6,8.4)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0.855,0.925)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderaxespad=0, prop=font1)  #显示上面的label

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}

plt.xlabel('The number of attention layers',font2) #x_label
plt.ylabel('Average $\mathregular{FDE_{20}}$',font2)#y_label
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()