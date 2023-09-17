
# In[1]:


import numpy as np
from collections import Counter
# from itertools import imap
from math import sqrt
from operator import mul
from sklearn.model_selection import train_test_split


#  L1, L2, cosine distance

def ecludian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1,x2):
    return np.sum(abs(x1-x2))


def cosine(x1,x2):
#     x1=x1.ravel()
#     x2=x2.ravel()
    return 1 - (np.dot(x1.ravel(),x2.ravel())/np.linalg.norm(x1.ravel())*np.linalg.norm(x2.ravel()))
#     return 1 - (np.sum(map(mul, x1, x2))/ np.sqrt(np.sum(map(mul, x1, x1))* np.sum(map(mul, x2, x2))))


class KNN:

    def __init__(self, k=3, distance_metric='euclidean', encoder='resnet'):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder = encoder
        self.X = None
        self.y =None
        self.X_train= None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        
    def train_test(self, data):
        if self.encoder =='resnet':
            self.X=data[:,1]
        if self.encoder =='vit':
            self.X=data[:,2]
        self.y=data[:,3]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.X , self.y, test_size=0.2, random_state=1234)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit( self, X ,  y ):
        self.X_train = X
        self.y_train = y
       
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def set_k(self, k):
        self.k = k
        
    def set_distance_metric(self, distance_metric):
        self.distance_metric = distance_metric
        

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self,x):
#         print(self.distance_metric)
        # compuete distances
        if (self.distance_metric=='euclidean'):
#             print(1)
            distances = [ecludian_distance(x, x_train) for x_train in self.X_train]
#             print(distances)
        
        if (self.distance_metric=='manhattan_distance'):
#             print(1)
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
#             print(distances)

        if (self.distance_metric=='cosine'):
#             print(1)
            distances = [cosine(x, x_train) for x_train in self.X_train]
#             print(distances)
            
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # most commomn labels
        most_common = Counter(k_nearest_labels).most_common(1) 
        
        return most_common[0][0]


# In[5]:


# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# data= sys.argv[1:]
# X,y =  data[:, 1] , data[:, 3]
# X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=1234)


# In[4]:


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def run(data):
#     X,y =  data[:, 1] , data[:, 3]
#     X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=1234)
    clf = KNN( k=4)
    # knn.set_distance_metric('manhattan_distance')
    X_train, X_test, y_train, y_test = clf.train_test(data)
    clf.fit(X_train, y_train)
    predictions= clf.predict(X_test)

#     acc = np.sum(predictions == y_test)/len(y_test)

    f1 = f1_score(y_test, predictions,average='macro')
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions,average='macro')
    recall = recall_score(y_test, predictions,average='macro')
    return {"accuracy": accuracy, 'Precision': precision, "Recall": recall, "F1": f1}


# In[7]:


import sys
file=sys.argv[1]

try:
    data=np.load(file, allow_pickle=True)
    
except Exception as e:
    print("Error in loading data")
    sys.exit(1)

print(run(data))


# In[ ]:




