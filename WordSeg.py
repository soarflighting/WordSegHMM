import sys
import math
import os

class SegWordHMM:
    def __init__(self,train_file_path,test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        # 每个状态所对应的字的个数
        self.state_word_count = dict()
        # 每个状态的个数
        self.state_count = dict()
        # 包含的字列表
        self.words = list()
        # 状态列表
        self.state = ['B','M','E','S']
        # 初始PI 的概率，第一个字不能是M、E 只能是B、S
        self.pi_dict = {'B':0.5,'M':0,'E':0,'S':0.5}
        # 状态转换矩阵
        self.A_dict = dict()
        # 观测矩阵
        self.B_dict = dict()

    # 返回字符在状态中的索引位置
    def findIndex(self,i,lens):
        if lens == 1:
            return 'S'
        if i == 0:
            return 'B'
        if i == lens-1:
            return 'E'
        return 'M'

    def training(self):
        print("training......")
        fpo = open(self.train_file_path,'r')

        strs = ""

        for line in fpo:
            if self.train_file_path.split('.')[-1] == "utf8":
                line = line.decode('utf-8')
            line = line.replace(' \n','')
            line = line.replace('\n','')
            grap = line.split('  ')
            for scen in grap:
                scen = scen.strip()
                lens = len(scen)
                for i in range(0,lens):
                    wd = scen[i]
                    if wd not in self.words:
                        self.words.append(wd)
                    # 得到字符对应的状态码
                    st = self.findIndex(i,lens)
                    self.state_count.setdefault(st,0)
                    self.state_count[st] += 1
                    strs += st
                    self.state_word_count.setdefault(st,{})
                    self.state_word_count[st].setdefault(wd,0)
                    self.state_word_count[st][wd] += 1
            strs += ','
        fpo.close()

        # 得到每个状态之后的状态的个数
        ast = dict()
        for i in range(len(strs) - 2):
            st1 = strs[i]
            st2 = strs[i+1]
            if st1 == ',' or st2 == ',':
                continue
            ast.setdefault(st1,{})
            ast[st1].setdefault(st2,0)
            ast[st1][st2] += 1

        # 构建转换矩阵
        # 初始化转换矩阵A_dict
        for st1 in self.state:
            self.A_dict.setdefault(st1,{})
            for st2 in self.state:
                self.A_dict[st1].setdefault(st2,0)
        for st1,item in ast.items():
            for st2 in item.keys():
                self.A_dict[st1][st2] = float(item[st2])/float(self.state_count[st1])

        # 构建观测矩阵
        # 初始化观测矩阵
        for st in self.state:
            self.B_dict.setdefault(st,{})
            for wd in self.words:
                self.B_dict[st].setdefault(wd,1.0/float(self.state_count[st]))
        for st,item in self.state_word_count.items():
            for wd in item.keys():
                self.B_dict[st][wd] = float(item[wd])/float(self.state_count[st])
        fpo.close()
        print('training completed ...')

    def testing(self):
        print("testing ...")
        fpo = open(self.test_file_path,'r')
        file_name = self.test_file_path.split('.')[:-1]
        file_name = "".join(file_name)+'_result.utf8'
        fpw = open(file_name,'w')

        fi = {}
        num = 0
        for eachline in fpo:
            num += 1
            if file_name[-1] == "utf8":
                line = eachline.decode('utf-8')
            line = eachline.strip()
            lens = len(line)
            if lens < 1:
                continue
            wd = line[0]
            for st in self.state:
                fi.setdefault(1,{})
                if wd not in self.B_dict[st].keys():
                    self.B_dict[st].setdefault(wd,1.0/float(self.state_count[st]))
                fi[1].setdefault(st,self.pi_dict[st]*self.B_dict[st][wd])
            for i in range(1,lens):
                wd = line[i]
                fi.setdefault(i+1,{})
                for st1 in self.state:
                    fi[i+1].setdefault(st1,0)
                    max_num = 0
                    for st2 in self.state:
                        max_num = max(max_num,fi[i][st2]*self.A_dict[st2][st1])
                    if wd not in self.B_dict[st1].keys():
                        self.B_dict[st1][wd] = 1.0/float(self.state_count[st1])
                    fi[i+1][st1] = max_num*self.B_dict[st1][wd]
            links = []
            tmp = []
            for st in self.state:
                tmp.append([st,fi[lens][st]])
            st1,_ = max(tmp,key = lambda x:x[1])

            for i in range(lens,1,-1):
                tmp = []
                for st in self.state:
                    tmp.append([st,fi[i-1][st]*self.A_dict[st][st1]])
                st1,sc = max(tmp,key = lambda x:x[1])
                links.append(st1)
            links.reverse()

            strs = ""
            for i in range(len(links)):
                st = links[i]
                if st == 'S':
                    strs += (line[i]+'  ')
                    continue
                if st == 'B' or st == 'M':
                    strs += line[i]
                    continue
                if st == 'E':
                    strs += (line[i]+'  ')
            strs += '\n'
            fpw.writelines(str(strs.encode('utf-8')))
            print(strs)
        fpo.close()
        fpw.close()
        print("分词完成，一共{}行".format(num))

if __name__ == '__main__':
    train_file_path = 'd:/input_data/icwb2-data/training/pku_training.txt'
    test_file_path = 'd:/input_data/icwb2-data/testing/pku_test.txt'
    segword = SegWordHMM(train_file_path,test_file_path)
    segword.training()
    segword.testing()

