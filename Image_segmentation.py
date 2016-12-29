import os
import numpy                                                                    
import random 
from PIL import Image                                                            
import matplotlib.pyplot as plt  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def get_5_image(filename):
	im_list = []
	for i in filename:
		print(i)
		im = Image.open(i)
		im_list = im_list+[im.resize((1000,1000))]
		h, w =im.size[0],im.size[1]
		if (w<1000 or h<1000):	
			im_list += [im.rotate(angle).resize((1000,1000)) for angle in random.sample(range(0,180),4)]
		if (w>1000 and h > 1000):
			im_list +=[im.crop((h2,w2,h2+1000,w2+1000)) for h2,w2 in zip(random.sample(range(0,h-1000),4),random.sample(range(0,w-1000),4))]
	return(im_list)

############### enlarge the database 
mag_filename=os.listdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Malignant")
ben_filename=os.listdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Benign")

mag_filename_train,mag_filename_test = mag_filename[0::2], mag_filename[1::2]
ben_filename_train,ben_filename_test = ben_filename[0::2], ben_filename[1::2]
os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Malignant")
mag_train = get_5_image(mag_filename_train)
X_test_mag =  [numpy.array(Image.open(i).convert('L').resize((1000,1000)), 'f').flatten() for i in mag_filename_test] 

os.chdir("/Users/menghanliu/Documents/Class/2016_fall_Medical_infroamtics/FinalProject_MalanomaImage/0_data/Benign")
ben_train = get_5_image(ben_filename_train[1:])
X_test_ben =[numpy.array(Image.open(i).convert('L').resize((1000,1000)), 'f').flatten() for i in ben_filename_test]

X_train = numpy.array([numpy.array(i.convert("L"),"f").flatten() for i in mag_train] + [numpy.array(i.convert("L"),"f").flatten() for i in ben_train])
y_train = numpy.array(["Malignant"]*len(mag_train)+["Benign"]*len(ben_train))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3,6,9), random_state=1)
clf.fit(X_train,y_train)

X_test= numpy.array(X_test_mag+ X_test_ben)
y_test =   numpy.array(["Malignant"]*len(mag_filename_test)+["Benign"]*len(ben_filename_test)) #resize to 1000 by 1000

confusion_matrix(y_test,clf.predict(X_test),labels=["Malignant","Benign"])

fpr, tpr, thresholds = roc_curve(y_test,clf.predict_proba(X_test)[:,1],pos_label="Malignant")
roc_auc = auc(fpr, tpr,)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.show()




