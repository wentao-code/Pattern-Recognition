# coding=utf-8
import numpy as np 
from pathlib import Path
#生成随机格式示例，不是正式模型输出
def store_txt(label,result_txt):
    with open(result_txt,"w") as w:
        for index,result in enumerate(label,1):#label后面的1,代表下标从1开始
            w.write(str(index)+" "+str(result)+"\n")#按网站要求格式写入，imageID（int）+" "（此处为空格非\t）+label（-1\1,int）+"\n"
def main():
    label = np.random.binomial(1, 0.5, 1000)#生成1000条随机示例标签，不代表测试集标签
    label[label==0] = -1 #标签映射到1,-1
    result_path = Path(__file__).resolve().with_name('task2_result_demo_random.txt')
    store_txt(label,result_path)#仅生成随机标签格式示例，不是模型预测结果
    print(f'Wrote sample format file: {result_path}')
if __name__ == "__main__":
    main()
