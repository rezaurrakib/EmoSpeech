import os
import scipy
import pickle
import sklearn
import speechpy
import numpy as np
import soundfile as sf
import scipy.io.wavfile
import matplotlib.pyplot as plt

from scipy import io
from sklearn import svm
from visualization import *
from pydub import AudioSegment
from sklearn.svm import LinearSVC
from pydub.utils import make_chunks
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

__author__     = "Reza"
__copyright__  = "Md Rezaur Rahman, TUM(Computer Science)"
__email__      = "reza.rahman@tum.de"
__maintainer__ = "Reza"
__status__     = "Dev"


FRAME_SIZE = 16000
MFCC_LEN = 39


# The class for training the classifier
# Only RAVDESS dataset has been used

root_folder = os.getcwd()
dataset_path = os.path.join(root_folder, "Dataset/RAVDESS/")


def calc_variance(data):
    N = 10
    mean = np.mean(data)
    sum = 0.0
    for i in range(10):
        sum += abs(data[i] - mean)**2
    s = float(sum/(N-1))
    return s

class RavdessEmoClassifier():
    def __init__(self):
        self.dataset_folder_path = dataset_path
        self.dataset_class_labels = ["Neutral", "Angry", "Happy", "Sad", "Calm", "Fearful", "Disgust", "Surprised"]
        #self.dataset_class_labels = ["Neutral", "Angry", "Happy", "Sad"] # Label for Berlin DB
        self.max_signal_length = 50000  # Hypertuned for RAVDESS dataset

    def show_info(self, aname, a):
        print("Array", aname)
        print("shape:", a.shape)
        print("dtype:", a.dtype)
        print("min, max:", a.min(), a.max())

    def extract_mfcc_feature(self):
        mfcc_dataset = []
        dataset_label = []

        print("dataset_folder_path: ", self.dataset_folder_path)
        print("=========== Commencing reading the RAVDESS Dataset ============")
        for i, directory in enumerate(self.dataset_class_labels):
            cnt = 0
            print("The Directory is : ", directory)
            os.chdir(self.dataset_folder_path + "/" + directory)

            for filename in os.listdir('.'):
                ## The Signal data is returned as a numpy array with a data-type determined from the file.
                data, sample_rate = sf.read(filename)
                #print("sample rate: ", sample_rate)
                # Checking whether data channel is stereo or mono
                if type(data[0]).__name__ == 'ndarray':
                    signal = data[:, 1]
                else:
                    signal = data

                # self.show_info("data", signal)
                # The signals are padded if it is less than required
                # Otherwise slice the signals
                signal_len = len(signal)
                pad_length = abs(self.max_signal_length - signal_len)
                pad_remainder = pad_length % 2
                pad_length = int(pad_length / 2)

                if (signal_len < self.max_signal_length):
                    signal = np.pad(signal, (pad_length, pad_length + pad_remainder), 'constant', constant_values=0)
                else:
                    signal = signal[pad_length:pad_length + self.max_signal_length]
                cnt = cnt + 1

                ## mfcc_len: Number of mfcc features to take for each frame
                mfcc_len = 39
                #mfcc = speechpy.feature.mfcc(signal, sample_rate, num_cepstral=mfcc_len)
                mfcc = speechpy.feature.mfcc(signal, 32000, num_cepstral=MFCC_LEN)
                mfcc = mfcc.flatten()
                #print("len(mfcc) : ", len(mfcc))
                mfcc_dataset.append(mfcc)
                dataset_label.append(i)  # Lebelling the mfcc feature

            print("Number of Samples for training in", directory, "directory is : ", cnt)

        return mfcc_dataset, dataset_label

    def training_feature_visualization(self, mfcc_dataset, dataset_label):
        features_embedding_visualize_3d(mfcc_dataset, dataset_label, root_folder)
        #features_embedding_visualize_2d(mfcc_dataset, dataset_label, root_folder)

    def train_test_val_split(self, mfcc_dataset, dataset_label):
        # Only train & test set
        x_train, x_test, y_train, y_test = train_test_split(mfcc_dataset, dataset_label, train_size=0.8, random_state=42)
        return x_train, x_test, y_train, y_test


    def run_model(self, model_name, x_train, y_train, x_test, y_test):
        model = None
        if model_name == "SVM_Linear":
            model = (svm.SVC(kernel='linear'))
        elif model_name == "LinSVC":
            model = model = LinearSVC(multi_class='crammer_singer')
        elif model_name == "SVM_RBF":
            model = (svm.SVC(kernel='rbf'))
        elif model_name == "Gauss_NB":
            model = GaussianNB()
        elif model_name == "RandomForest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "MLP":
            model = MLPClassifier(activation='logistic', verbose=True, hidden_layer_sizes=(512,), batch_size=32)

        clf = model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)
        acc = accuracy_score(y_pred=y_prediction, y_true=y_test)
        return acc

    def k_fold_cross_validation(self, mfcc_dataset, dataset_label, model_name):
        print("===== Result For Model ", model_name, " =====")
        mfcc_dataset = np.array(mfcc_dataset)
        dataset_label = np.array(dataset_label)
        acc_list = []
        # 10-fold cross validation
        n_folds = 10
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for fold in range(0, n_folds):
            cv_splits = list(skf.split(mfcc_dataset, dataset_label))
            train_indices = cv_splits[fold][0]
            test_indices = cv_splits[fold][1]

            #print(len(train_indices))
            #print("train indices: ", train_indices)
            #print(len(test_indices))
            #print("test indices: ", test_indices)

            x_train = mfcc_dataset[train_indices]
            y_train = dataset_label[train_indices]

            x_test = mfcc_dataset[test_indices]
            y_test = dataset_label[test_indices]
            accuracy = self.run_model(model_name, x_train, y_train, x_test, y_test)
            acc_list.append(accuracy)
            print("The Accuracy score in fold", str(fold + 1), "is: ", accuracy)

        print("Accuracy List : ", acc_list)
        avg_acc = np.mean(acc_list)
        variance = calc_variance(acc_list)
        print("Avg acc: ", avg_acc)
        print("Variance: ", variance)

    def display_metrics(self, y_pred, y_test):
        print("The Accuracy score is : ")
        print(accuracy_score(y_pred=y_pred, y_true=y_test))
        print("The Confusion Matrix is : ")
        print(confusion_matrix(y_pred=y_pred, y_true=y_test))

    def training_with_test_train_split(self, mfcc_dataset, dataset_label, model_name):
        x_train, x_test, y_train, y_test = self.train_test_val_split(mfcc_dataset, dataset_label)
        accuracy = self.run_model(model_name, x_train, y_train, x_test, y_test)
        print("The Accuracy score for ", model_name, " model is: ", accuracy)



if __name__ == "__main__":
    model = RavdessEmoClassifier()
    mfcc_dataset, dataset_label = model.extract_mfcc_feature()
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "RandomForest")
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "SVM_Linear")
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "SVM_RBF")
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "Gauss_NB")
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "LinSVC")
    model.k_fold_cross_validation(mfcc_dataset, dataset_label, "MLP")

    '''model.training_with_test_train_split(mfcc_dataset, dataset_label, "RandomForest")
    model.training_with_test_train_split(mfcc_dataset, dataset_label, "SVM_Linear")
    model.training_with_test_train_split(mfcc_dataset, dataset_label, "SVM_RBF")
    model.training_with_test_train_split(mfcc_dataset, dataset_label, "Gauss_NB")
    model.training_with_test_train_split(mfcc_dataset, dataset_label, "LinSVC")
    model.training_with_test_train_split(mfcc_dataset, dataset_label, "MLP")'''
