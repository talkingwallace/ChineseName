import pandas as pd
import numpy as np

# 0 female 1 male
# 83%左右的准确率
def dataPreprocessing(df):

    value = list(df[df.columns[0]])
    id = []
    name = []
    gender = []
    for i in value:
        s = i.split(',')
        id.append(s[0])
        name.append(s[1])
        try:
            gender.append(s[2])
        except:
            pass

    df = pd.DataFrame()
    df['id'] = id
    df['name'] = name
    if len(gender)!= 0:
        df['gender'] = gender
    return df

data = dataPreprocessing(pd.read_table(r'./data/train.txt'))
test = dataPreprocessing(pd.read_table(r'./data/test.txt'))

valid = data[-20000:-1]
data = data[0:100000]

class NaiveBayes():

    # 这个贝叶斯分类器仅仅适用于当前的数据

    def __init__(self):
        self.Ppos = 0
        self.Pneg = 0
        self.lookUpTable = pd.DataFrame()

    def handleName(self,name):
        name = name.apply(lambda x: x + ' ' if len(x) < 2 else x)
        return name.apply(lambda x: [x[0], x[1]])

    def fit(self,data,classLabel = 'gender'):
        data[classLabel] = data[classLabel].apply(lambda x:int(x))
        c = np.array(data[classLabel])
        self.data = data
        self.Ppos = len(c.nonzero()[0])/len(c)
        self.Pneg = 1 - self.Ppos

        self.features = self.handleName(data['name'])
        # compute p(x|0) and p(x|1)
        self.vocab = set()
        for i in self.features:
            for k in i:
                self.vocab.add(k)
        self.vocab = list(self.vocab)
        Dict = dict(zip(self.vocab,[i for i in range(len(self.vocab))]))
        posCount = np.zeros(len(self.vocab))
        negCount = np.zeros(len(self.vocab))

        for f,c in zip(self.features,data[classLabel]):
            if c == 0:
                tmp = negCount
            else:
                tmp = posCount

            tmp[Dict[f[0]]]+=1
            tmp[Dict[f[1]]]+=1

        self.lookUpTable['feature'] = list(Dict.keys())
        self.lookUpTable['pos'] = posCount
        self.lookUpTable['neg'] = negCount

        # 拉普拉斯平滑
        self.lookUpTable['pos'] += 1
        self.lookUpTable['neg'] += 1

        # 转换成概率
        self.lookUpTable['pos'] = self.lookUpTable['pos'] / (len(data)*self.Ppos)
        self.lookUpTable['neg'] = self.lookUpTable['neg'] / (len(data)*self.Pneg)
        self.Dict = Dict


    def predict_one_sample(self,features=[]):
        for id,i in enumerate(features):
            if i not in self.Dict:
                features[id] = ' '

        a = self.lookUpTable.loc[self.Dict[features[0]]]
        b = self.lookUpTable.loc[self.Dict[features[1]]]

        scoreA = np.log(a.pos) + np.log(b.pos) + np.log(self.Ppos)
        scoreB = np.log(a.neg) + np.log(b.neg) + np.log(self.Pneg)

        return 1 if scoreA>scoreB else 0


    def predict_batch(self,data):
        f = self.handleName(data['name'])
        rs = []
        for i in f:
            rs.append(self.predict_one_sample(i))
        return rs

nv = NaiveBayes()
nv.fit(data)
rs = nv.predict_batch(test)
test['gender'] = rs
test = test.drop(columns = ['name'])
test.to_csv('result.csv',index=False)
