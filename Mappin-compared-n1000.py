import csv
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import savgol_filter
import  re
from scipy import stats

def read_csv(files: object) -> object:
    """读取csv文件"""
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    # 读取csv文件中2,3列的数据，且转化为float类型
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y


def write_csv(files: object, x01, y01) -> object:
    with open(files, 'w', newline='') as csvfile:
        plots = csv.writer(csvfile, delimiter=',')
        z = np.array(x01) - np.array(y01)
        for i in range(len(z)):
            set01 = [i, i, z[i]]
            plots.writerow(set01)


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

plt.figure(1)

plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

x1_1, y1_1 = read_csv(
    "100users/100csv_drn.csv")
# y1_1 = savgol_filter(y1_1, 5, 3)
plt.plot(x1_1, y1_1, color='coral', label='RB-DRN', linewidth=1.5)

mean, std = np.mean(y1_1[79:]), np.std(y1_1[79:])
conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)#计算95%的上下区间
y1_1a=y1_1-(conf_intveral[1]-conf_intveral[0])/2
y1_1b=y1_1+(conf_intveral[1]-conf_intveral[0])/2
plt.fill_between(x1_1, y1_1a, y1_1b, color='salmon', alpha=0.2)#填充区间


x1_0, y1_0 = read_csv(
    "100users/100csv_dqn.csv")
y1_0 = [element - 3 for element in y1_0 ]
# y1_0 = savgol_filter(y1_0, 5, 3)
plt.plot(x1_0, y1_0, color='steelblue', label='PDSU', linewidth=1.5)

mean, std = np.mean(y1_0[79:]), np.std(y1_0[79:])
conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)#计算95%的上下区间
y1_0a=y1_0-(conf_intveral[1]-conf_intveral[0])/2
y1_0b=y1_0+(conf_intveral[1]-conf_intveral[0])/2
plt.fill_between(x1_0, y1_0a, y1_0b, color='deepskyblue', alpha=0.2)#填充区间

x1_3, y1_3 = read_csv(
    "100users/100csv_greedy.csv")
# y1_3 = savgol_filter(y1_3, 5, 3)
plt.plot(x1_3, y1_3, color='seagreen', label='Greedy', linewidth=1.5)

mean, std = np.mean(y1_3[99:]), np.std(y1_3[99:])
conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)#计算95%的上下区间
y1_3a=y1_3-(conf_intveral[1]-conf_intveral[0])/3
y1_3b=y1_3+(conf_intveral[1]-conf_intveral[0])/3
plt.fill_between(x1_3, y1_3a, y1_3b, color='lightgreen', alpha=0.2)#填充区间

x1_2, y1_2 = read_csv(
    "100users/100csv_random.csv")
# y1_2 = savgol_filter(y1_2, 5, 3)
plt.plot(x1_2, y1_2, color='brown', label='Random', linewidth=1.5)

mean, std = np.mean(y1_1[79:]), np.std(y1_1[79:])
conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)#计算95%的上下区间
y1_2a=y1_2-(conf_intveral[1]-conf_intveral[0])/5
y1_2b=y1_2+(conf_intveral[1]-conf_intveral[0])/5
plt.fill_between(x1_2, y1_2a, y1_2b, color='chocolate', alpha=0.2)#填充区间






plt.xlabel('Number of training steps (×10$^5$)',  fontsize=15)
plt.ylabel('Average system utility ', fontsize=15)
# plt.title('Number of users (N = 15)')
# plt.legend()
plt.legend(bbox_to_anchor=(0.35, 1, 0.3, 0.15), loc='center',
           fontsize=12, ncol=4, columnspacing=0.5)
plt.xlim(0, 150)
plt.xticks([0,30,60,90,120,150], ['0', '0.6', '1.2', '1.8','2.4','3'])
# plt.ylim(0.5, 3.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

matplotlib.pyplot.grid(b=True, which='major', axis='both', )
plt.tight_layout()


plt.show()
