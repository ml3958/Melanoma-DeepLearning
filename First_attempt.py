import os
from PIL import Image                                                            
import numpy                                                                     
import matplotlib.pyplot as plt  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/")


mag_filename=os.listdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Malignant")
ben_filename=os.listdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Benign")
#print(filename)
#mag_im_array = [numpy.array(Image.open(i).convert('L'), 'f') for i in temp]  #black and white
os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Malignant")
mag_im_array_1000 = [numpy.array(Image.open(i).convert('L').resize((256,256)), 'f').flatten() for i in mag_filename]   #resize to 1000 by 1000
mag_im_array_1000 = numpy.array(mag_im_array_1000)

os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Benign")
ben_im_array_1000 = [numpy.array(Image.open(i).convert('L').resize((256,256)), 'f').flatten() for i in ben_filename[1:]]   #resize to 1000 by 1000
ben_im_array_1000 = numpy.array(ben_im_array_1000)


X = numpy.concatenate((mag_im_array_1000,ben_im_array_1000))
y = numpy.array(["Malignant"]*len(mag_im_array_1000)+["Benign"]*len(ben_im_array_1000))

X_train, X_test = X[0::2], X[1::2]
y_train, y_test = y[0::2], y[1::2]

clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(50,50), learning_rate="adaptive")
clf.fit(X_train,y_train)



#confusion matrix
y_predict=clf.predict(X_test)
confusion_matrix(y_test,clf.predict(X_test),labels=["Malignant","Benign"])


fpr, tpr, thresholds = roc_curve(y_test2,clf2.predict_proba(X_test2)[:,1],pos_label="Malignant")
roc_auc = auc(false_positive_rate, true_positive_rate,)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
 label = label_string + ' (AUC = %0.3f)' % roc_auc)
