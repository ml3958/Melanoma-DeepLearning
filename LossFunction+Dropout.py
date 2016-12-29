import os,random
from PIL import Image                                                            
import numpy                                                                     
import matplotlib.pyplot as plt  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc,log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sknn.mlp import Classifier, Layer,MultiLayerPerceptron



os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/")
X_test = numpy.load("X_test.npy")
y_test = numpy.load("y_test.npy")
X_train = numpy.load("X_train.npy")
y_train = numpy.load("y_train.npy")

        


loss=[]
loss2=[]

for Layerdepth in range(1,3):
	for Neuron in range(5,26,5):
		Layers = [Layer("Rectifier", units=Neuron)]*Layerdepth + [Layer("Softmax")]
		nn = Classifier(
    			layers=Layers,
			    learning_rate=0.001,
				n_iter=30,        
				dropout_rate=0.25,
				loss_type="mcc",
				random_state=1)
		nn.fit(X_train_1, y_train_1)
		nn2 = Classifier(
    			layers=Layers,
		    	learning_rate=0.001,
				n_iter=30,        
				dropout_rate=0.25,
				loss_type="mcc",
				random_state=2)
		nn2.fit(X_train_1, y_train_1)		
		loss += [log_loss(y_train_2.tolist(),nn.predict_proba(X_train_2))]
 		loss2 += [log_loss(y_train_2.tolist(),nn2.predict_proba(X_train_2))]




confusion_matrix(y_test,nn.predict(X_test),labels=["Malignant","Benign"])

fpr, tpr, thresholds = roc_curve(y_test,nn.predict_proba(X_test)[:,1],pos_label="Malignant")
roc_auc = auc(fpr, tpr,)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.show()


