import sys
from time import time
import matplotlib.pyplot as plt
sys.path.append("C:\\Users\\satyam\\Desktop\\MajorProject Final\\Emails-Classification-UsingSupervisedLeraningTechniques\\")
from email_preprocess import preprocess
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#using the Gaussian Bayes algorithm for classification of emails.
#the algorithm is imported from the sklearn library
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#initializaing the test and train features and labels
#the function preprocess is imported from email_preprocess.py

# features_train, features_test, labels_train, labels_test = preprocess()

def Naive_Bayes():
    # defining the classifier
    clf = GaussianNB()

    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"Naive_Bayes": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"Naive_Bayes": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")

    # calculating and printing the accuracy
    print("Accuracy of Naive Bayes: ", accuracy_score(pred, labels_test),"\n")
    accuracy.update({"Naive_Bayes":accuracy_score(pred, labels_test)})
# Naive_Bayes()
def Support_Vector_Machine():
    # defining the classifier
    clf = SVC(kernel='linear', C=1)

    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"Support_Vector_Machine": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"Support_Vector_Machine": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")
    accuracy.update({"Support_Vector_Machine": accuracy_score(pred, labels_test)})
    # calculating and printing the accuracy of the algorithm
    print("Accuracy of SVM Algorithm: ", clf.score(features_test, labels_test),"\n")
# Support_Vector_Machine()
def Decision_Trees():
    # defining the classifier
    clf = tree.DecisionTreeClassifier()

    print("\nLength of Features Train", len(features_train[0]))

    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"Decision_Trees": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"Decision_Trees": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")
    accuracy.update({"Decision_Trees": accuracy_score(pred, labels_test)})
    # calculating and printing the accuracy of the algorithm
    print("Accuracy of Decision Trees Algorithm: ", accuracy_score(pred, labels_test),"\n")
# Decision_Trees()
def AdaBoost_Classfier():
    # defining the classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)

    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"AdaBoost_Classfier": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"AdaBoost_Classfier": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")
    accuracy.update({"AdaBoost_Classfier": accuracy_score(pred, labels_test)})
    # calculating and printing the accuracy of the algorithm
    print("Accuracy of Ada Boost Classifier: ", accuracy_score(pred, labels_test),"\n")
# AdaBoost_Classfier()
def K_Nearest_Neigbhour():
    # defining the classifier
    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"K_Nearest_Neigbhour": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"K_Nearest_Neigbhour": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")
    accuracy.update({"K_Nearest_Neigbhour": accuracy_score(pred, labels_test)})
    # calculating and printing the accuracy of the algorithm
    print("Accuracy of KNN Algorithm: ", accuracy_score(pred, labels_test),"\n")
# K_Nearest_Neigbhour()
def Random_forest():

    # defining the classifier
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    # predicting the time of train and testing
    t0 = time()
    clf.fit(features_train, labels_train)
    TrainingTime.update({"K_Nearest_Neigbhour": round(time() - t0, 3)})
    print("\nTraining time:", round(time() - t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    PredictionTime.update({"K_Nearest_Neigbhour": round(time() - t1, 3)})
    print("Predicting time:", round(time() - t1, 3), "s")
    accuracy.update({"Random_forest": accuracy_score(pred, labels_test)})
    # calculating and printing the accuracy of the algorithm
    print("Accuracy of Random Forest Algorithm: ", accuracy_score(pred, labels_test),"\n")
# Random_forest()

if __name__=="__main__":
    accuracy = {}
    TrainingTime={}
    PredictionTime={}
    features_train, features_test, labels_train, labels_test = preprocess()
    print("\nChoose the algorithm to use:")
    print("1.Naive Bayes.")
    print("2.Support Vector Machine.")
    print("3.Decision Trees.")
    print("4.AdaBoost Classifier.")
    print("5.K Nearest Neigbhour.")
    print("6.Random Forrest.")
    print("0.Run with all Algorithms.\n")
    state='y'
    while(state=="y"):
        print("Enter the number to apply: ",end="")
        numb=int(input())

        if numb==1:
            Naive_Bayes()
        elif numb ==2:
            Support_Vector_Machine()
        elif numb==3:
            Decision_Trees()
        elif numb==4:
            AdaBoost_Classfier()
        elif numb==5:
            K_Nearest_Neigbhour()
        elif numb==6:
            Random_forest()
        elif numb==0:
            Naive_Bayes()
            Support_Vector_Machine()
            Decision_Trees()
            AdaBoost_Classfier()
            K_Nearest_Neigbhour()
            Random_forest()
            # accuracy={}
            # accuracy = {"Naive_Bayes": a, "Support_Vector_Machine": b, "Decision_Trees": c, "AdaBoost_Classfier": d,
            #             "K_Nearest_Neigbhour": e, "Random_forest": f}
            print(accuracy)
            print(TrainingTime)
            print(PredictionTime)

            Algorithms = accuracy.keys()
            Accuracy_Scores = accuracy.values()
            plt.bar(Algorithms, Accuracy_Scores)
            plt.title(" Algorithms with accuracy ")
            plt.xlabel("Algorithms")
            plt.ylabel("Accuracy_Scores")
            plt.show()

            Algorithms_Train = TrainingTime.keys()
            Training_Scores = TrainingTime.values()
            plt.bar(Algorithms_Train, Training_Scores)
            plt.title(" Algorithms with Training_Scores ")
            plt.xlabel("Algorithms")
            plt.ylabel("Training_Scores")
            plt.show()

            Algorithms_Predict = PredictionTime.keys()
            Prediction_Scores = PredictionTime.values()
            plt.bar(Algorithms_Predict, Prediction_Scores)
            plt.title(" Algorithms with Prediction_Scores ")
            plt.xlabel("Algorithms")
            plt.ylabel("Prediction_Scores")
            plt.show()


        else:
            print("Invalid Number /nEnter Valid Number")
        print("do you want to continue with another alogorithm(y/n): ",end="")
        state = input()


# {'Naive_Bayes': 0.9732650739476678, 'Support_Vector_Machine': 0.9840728100113766, 'Decision_Trees': 0.9897610921501706, 'AdaBoost_Classfier': 0.9664391353811149, 'K_Nearest_Neigbhour': 0.8572241183162684, 'Random_forest': 0.7775881683731513}
# {'Naive_Bayes': 1.437, 'Support_Vector_Machine': 100.494, 'Decision_Trees': 66.294, 'AdaBoost_Classfier': 189.51, 'K_Nearest_Neigbhour': 3.171}
# {'Naive_Bayes': 0.219, 'Support_Vector_Machine': 10.374, 'Decision_Trees': 0.033, 'AdaBoost_Classfier': 2.921, 'K_Nearest_Neigbhour': 0.063}