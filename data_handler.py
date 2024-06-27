# -*- coding: utf-8 -*-
import time

# liac-arff
import arff
import pickle
from pandas import DataFrame
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from preprocessing import LoadDriftData
from config import args
from collections import Counter
import os

class DriftDataset():
    """stream learning dataset loading class"""

    _datasetDict = {
        # RealData_UnknDrift original_data sets
        "elec": "2_RealData_UnknDrift/original_data/elecNorm.arff",
        "weat": "2_RealData_UnknDrift/original_data/weather_NSE.arff",
        "spam": "2_RealData_UnknDrift/original_data/spam_corpus_x2_feature_selected.arff",
        "airl": "2_RealData_UnknDrift/original_data/airline.arff",
        "covt-binary": "2_RealData_UnknDrift/original_data/covtypeNormBinary.arff",
        "poke-binary": "2_RealData_UnknDrift/original_data/PokerHandBinary.arff",


        # SyntData_SyntDrift original_data sets
        "SEAa0": "1_SyntData_SyntDrift/original_data/SEAa0.arff",
        # "SEAg": "1_SyntData_SyntDrift/SEAg0.arff",
        "HYPi": "1_SyntData_SyntDrift/original_data/HYP0.arff",
        "AGRa": "1_SyntData_SyntDrift/original_data/AGRa4.arff",
        # "AGRg": "1_SyntData_SyntDrift/AGRg0.arff",
        # "LEDa": "1_SyntData_SyntDrift/LEDa0.arff",
        # "LEDg": "1_SyntData_SyntDrift/LEDg0.arff",
        "RBFi": "1_SyntData_SyntDrift/original_data/RBF0.arff",
        "RTGn": "1_SyntData_SyntDrift/original_data/RTG0.arff"
    }
    # for i in range(6):
    #     _datasetDict['AGRa' + str(i)] = '1_SyntData_SyntDrift/original_data/AGRa{}.arff'.format(i)
    #     _datasetDict['HYP' + str(i)] = '1_SyntData_SyntDrift/original_data/HYP{}.arff'.format(i)
    #     _datasetDict['RBF' + str(i)] = '1_SyntData_SyntDrift/original_data/RBF{}.arff'.format(i)
    #     _datasetDict['RTG' + str(i)] = '1_SyntData_SyntDrift/original_data/RTG{}.arff'.format(i)
    #     _datasetDict['SEAa' + str(i)] = '1_SyntData_SyntDrift/original_data/SEAa{}.arff'.format(i)

    def __str__(self):
        return "Class: concept_drift_dataset_loader"

    def __init__(self, path, dataset_name):
        self.DATA_FILE_PATH = path
        self.load_np(dataset_name)

    def load_np(self, dataset_name):
        self.file_ = self.DATA_FILE_PATH + self._datasetDict[dataset_name]
        # data, meta = arff.loadarff(self.file_)
        # dataset = pd.DataFrame(data)
        dataset = arff.load(open(self.file_), encode_nominal=True)
        # dataset = arff.load(open(self.file_))
        # dataset = arff.loadarff(file_)
        # df = pd.DataFrame(dataset[0])
        # a = np.array(dataset["original_data"])
        # b = np.array(dataset["original_data"])
        # b = np.array(dataset["original_data"])
        # data_list = list(dataset)
        self.np_data = np.array(dataset['data'])

        datas = []
        for data in self.np_data:
            if data[-1] < 2:
                datas.append(data)
        self.np_data = np.array(datas)

        self.np_data=self.np_data.repeat(1,axis=0)
        print(self.np_data)


def generation_data(filename, stream):
    naive_bayes = GaussianNB()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = 765  # Detect the type every 765 points.
    window_len = 15  # Every 15 errors form a data point.
    data_count = stream.shape[0] // max_samples

    all_data = []
    while n_samples < max_samples * data_count:
        resultList = []
        iter_max_samples = max_samples / window_len
        iter_n_samples = 0
        # Train the estimator with the samples provided by the original_data stream
        while iter_n_samples < iter_max_samples:
            correct_cnt = 0
            iter_n_window_samples = 0

            while iter_n_window_samples < window_len:
                X = stream[n_samples, :-1]
                X = np.expand_dims(X, 0)
                y = stream[n_samples, -1:]
                if iter_n_samples == 0 and iter_n_window_samples == 0:
                    sampling_zero = stream[np.where(stream[:, -1:] == 0)[0][:3]]
                    sampling_one = stream[np.where(stream[:, -1:] == 1)[0][:3]]
                    sampling_data = np.concatenate([sampling_zero, sampling_one], axis=0)
                    naive_bayes.fit(sampling_data[:, :-1], sampling_data[:, -1:])
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                naive_bayes.partial_fit(X, y)
                iter_n_window_samples = iter_n_window_samples + 1
                n_samples += 1
            iter_n_samples = iter_n_samples + 1

            resultList.append([correct_cnt / iter_n_window_samples])
        all_data.append(resultList)

    # Splitting into testing and training
    Ratio = 0.2  #  The proportion of the training set
    all_data = np.array(all_data).squeeze()
    train = all_data[0:int(data_count * Ratio), :]
    test = all_data[int(data_count * Ratio):, :]
    file_name = filename + '/train.csv'
    DataFrame(train).to_csv(file_name)
    file_name = filename + '/test.csv'
    DataFrame(test).to_csv(file_name)
    print("incremental processing has been completed")


def dataset_clear_train_test(file, dataset_name_list):
    for name in dataset_name_list:
        if not os.path.exists(file + '/pretrain_data/' + name):
            os.makedirs(file + '/pretrain_data/' + name)
        time.sleep(2)
        data_file_list = os.listdir(file + '/pretrain_data/' + name)
        if len(data_file_list) == 0:
            stream = DriftDataset('./datasets/', name).np_data
            generation_data(file + '/pretrain_data/' + name, stream)


def init_generation_dataset(args):
    # Reading the original_data
    print('Reading the train original_data')
    all_data_frame = LoadDriftData(args.Data_Vector_Length, args.DRIFT_FILE, args.DATA_SAMPLE_NUM)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    return Drift_data_array

def find_drift_point_mse(Drift_data_array):
    reference_data_x = Drift_data_array[:, :-2]
    reference_data_y = Drift_data_array[:, -2:-1]
    reference_data_z = Drift_data_array[:, -1:]
    for name in dataset_name_list:
        single_data_frame = pd.read_csv(os.path.join(file + '/pretrain_data/' + name, 'train.csv'), sep=',', header=0)
        data = single_data_frame.iloc[:, 1:]
        shift_data = single_data_frame.iloc[:, 1:].shift(-1, axis=1)
        shift_data_1 = shift_data - single_data_frame.iloc[:, 1:]
        train_error_feature = (shift_data_1 / data).iloc[:, :-1]

        single_data_frame = pd.read_csv(os.path.join(file + '/pretrain_data/' + name, 'test.csv'), sep=',', header=0)
        data = single_data_frame.iloc[:, 1:]
        shift_data = single_data_frame.iloc[:, 1:].shift(-1, axis=1)
        shift_data_1 = shift_data - single_data_frame.iloc[:, 1:]
        test_error_feature = (shift_data_1 / data).iloc[:, :-1]
        error_feature = pd.concat([train_error_feature, test_error_feature], ignore_index=True)

        # statistic = {"0": [], "1": [], "2": [], "3": []}
        statistic = []
        for feature in error_feature.to_numpy():
            mae_list = np.mean((feature - reference_data_x) ** 2, axis=1)
            _type = int(reference_data_y[np.argmin(mae_list)][0])
            _point = int(reference_data_z[np.argmin(mae_list)][0])
            statistic.append(_point)

        counter = Counter(statistic)
        most_common = counter.most_common(10)
        new_statistic = [common[0]for common in most_common]

        # new_statistic = set(statistic)
        with open(file + '/pretrain_data/' + name + '/point_statistic.pkl', 'wb') as f:
            pickle.dump(new_statistic, f)






if __name__ == "__main__":
    file = "./datasets/1_SyntData_SyntDrift"
    dataset_name_list = ['SEAa0', 'HYPi', 'AGRa', 'RTGn', 'RBFi']

    # file = "./datasets/2_RealData_UnknDrift"
    # dataset_name_list = ['elec', 'weat', 'spam', 'airl', 'poke-binary']

    Drift_data_array = init_generation_dataset(args)
    # Prepare the data into test set and training set.
    dataset_clear_train_test(file, dataset_name_list)

    # Calculate the average absolute error of drift points
    find_drift_point_mse(Drift_data_array)

        
