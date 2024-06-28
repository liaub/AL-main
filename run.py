import numpy as np
import torch
import os
import time
import pandas as pd
from config import args
from strategies.least_confidence import LeastConfidence
from preprocessing import LoadRealDriftData, LoadDriftData
from JointNetDrift import Joint_Prediction
from JointNetDriftVer import Joint_PredictionVer
def main():
    # TODO: 1. Based on model confidence, output the data that needs to be labeled.
    if not os.path.exists(args.DATA_FILE + 'mark'):
        os.mkdir(args.DATA_FILE + 'mark')
    train_x, train_y, train_locy, test_x, test_y, test_locy = init_supple_dataset(args)
    supple_data = [train_x, train_y, train_locy, test_x, test_y, test_locy]

    # Read fine-tuning data
    train_x, original_train_data, test_x, original_test_data = init_pretrain_dataset(args)
    # Load the network model
    net = Joint_PredictionVer(use_gpu=False, Data_Vector_Length=args.Data_Vector_Length, ModelSelect=args.FAN)
    BASE_PATH = args.BASE_PATH
    net.load_state_dict(torch.load(args.BASE_PATH + '/FAN_model_embeding.pkl'))
    # Filter strategy
    mark_num = 1
    strategy = LeastConfidence(train_x, None, None, original_train_data, net, args)
    # strategy.predict(BASE_PATH)  # Type-LDA accuracy
    need_mark_data = strategy.query(BASE_PATH)
    save_need_mark_data(args, mark_num, need_mark_data)

    while True:
        flag = input("Run mark_data.py to mark data. After completion, press 'Y' to fine-tune the model, or press any other key to prepare to exit the program:")
        if flag == "Y" or flag == "y":
            # TODO: 2. Manually annotate data with low confidence and proceed with model fine-tuning.
            original_data, inputs, label_y, label_l = read_mark_data(args)
            # Load the network model
            net = Joint_Prediction(use_gpu=False, Data_Vector_Length=args.Data_Vector_Length, ModelSelect=args.FAN)
            BASE_PATH = args.BASE_PATH
            net.load_state_dict(torch.load(BASE_PATH + '/FAN_model_embeding.pkl'))
            strategy = LeastConfidence(inputs, label_y, label_l, original_data, net, args)
            # training network model
            strategy.train(supple_data)

            # TODO: 3.Test the model on a real dataset.
            net = Joint_PredictionVer(use_gpu=False, Data_Vector_Length=args.Data_Vector_Length, ModelSelect=args.FAN)
            BASE_PATH = args.pretrain_model_path
            net.load_state_dict(torch.load(BASE_PATH + '/FAN_model_embeding.pkl'))
            strategy = LeastConfidence(train_x, None, None, original_train_data, net, args)
            strategy.predict(BASE_PATH)

            print("One round of work completed.===========================================")
            # TODO: 4.Re-screening data with low confidence.
            net = Joint_PredictionVer(use_gpu=False, Data_Vector_Length=args.Data_Vector_Length, ModelSelect=args.FAN)
            BASE_PATH = args.BASE_PATH
            net.load_state_dict(torch.load(BASE_PATH + '/FAN_model_embeding.pkl'))
            strategy = LeastConfidence(train_x, None, None, original_train_data, net, args)
            need_mark_data = strategy.query(BASE_PATH)
            save_need_mark_data(args, mark_num, need_mark_data)

        else:
            flag = input("Would you like to exit the program? Y/N:")
            if flag == "Y" or flag == "y":
                break


def read_mark_data(args):
    '''
    Read the label after the mark
    '''
    labels = []
    inputs = None
    original_data = None
    all_list = os.listdir(args.DATA_FILE + 'mark')
    for idx, file_name in enumerate(all_list):
        if '[' not in file_name:
            continue
        start_point = file_name.find('[')
        end_point = file_name.find(']')
        labels.append(eval(file_name[start_point: end_point+1]))
        single_file = file_name[0: start_point] + '_.csv'
        single_data_frame = pd.read_csv(os.path.join(args.DATA_FILE + 'mark', single_file), sep=',', header=0)
        shift_data = single_data_frame.iloc[:, -1].shift(-1)
        _shift_data = shift_data - single_data_frame.iloc[:, -1]
        error_feature = (_shift_data / single_data_frame.iloc[:, -1]).iloc[:-1]
        error_feature = pd.DataFrame(error_feature.values.reshape(1, -1))
        single_data_frame_T = pd.DataFrame(single_data_frame.iloc[:, -1].values.reshape(1, -1))
        if idx > 0:
            inputs = pd.concat([inputs, error_feature], ignore_index=True)
            original_data = pd.concat([original_data, single_data_frame_T], ignore_index=True)
        else:
            inputs = error_feature
            original_data = single_data_frame_T


    return original_data, inputs.to_numpy(), np.array(labels)[:, 0], np.array(labels)[:, 1]

def save_need_mark_data(args, mark_number, need_mark_data):
    '''
    Mark_Number: Number of marks;
    need_mark_data: Filtered original_data that needs to be labeled.
    '''
    all_list = os.listdir(args.DATA_FILE + 'mark')
    for name in all_list:
        os.remove(args.DATA_FILE + 'mark/' + name)
    time.sleep(1)
    for idx, data in enumerate(need_mark_data):
        flag_name = str(mark_number) + '_' + str(idx)
        path = args.DATA_FILE + 'mark/' + flag_name+'_'+'.csv'
        df = pd.DataFrame(data)
        # 保存 dataframe
        df.to_csv(path)

    print("The original_data that needs to be labeled has been filtered and saved at: {}. Please label it as soon as possible!".format(path))

def init_pretrain_dataset(args):
    # Reading the original_data
    print('Reading the pretrain_data original_data')
    Pretrain_Data_File = args.DATA_FILE
    # Data_Sample_Num = 100
    train_data, train_original, test_data, test_original = LoadRealDriftData(Pretrain_Data_File)
    Drift_data_array = train_data.values
    original_train_data = train_original.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))
    # random shuffle datasets
    # np.random.shuffle(Drift_data_array_train)
    # data_count = Drift_data_array_train.shape[0]  # original_data count
    train_x = Drift_data_array[:, 0:args.Data_Vector_Length]
    # train_y = Drift_data_array[:, -2]
    # train_l = Drift_data_array[:, -1]


    #test
    Drift_data_array = test_data.values
    original_test_data = test_original.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))
    # random shuffle datasets
    # np.random.shuffle(Drift_data_array_test)
    # data_count = Drift_data_array_test.shape[0]  # original_data count
    test_x = Drift_data_array[:, 0:args.Data_Vector_Length]
    # test_y = Drift_data_array[:, -2]
    # test_l = Drift_data_array[:, -1]
    return train_x, original_train_data, test_x, original_test_data


def init_supple_dataset(args):
    # Reading the original_data
    print('Reading the train original_data')
    all_data_frame = LoadDriftData(args.Data_Vector_Length, args.DRIFT_FILE, args.DATA_SAMPLE_NUM)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    # random shuffle datasets
    np.random.shuffle(Drift_data_array)
    data_count = Drift_data_array.shape[0]  # original_data count
    train_x = Drift_data_array[0:int(data_count * args.Train_Ratio), 0:args.Data_Vector_Length]
    train_y = Drift_data_array[0:int(data_count * args.Train_Ratio), -2]
    train_locy = Drift_data_array[0:int(data_count * args.Train_Ratio), -1]
    test_x = Drift_data_array[int(data_count * args.Train_Ratio):, 0:args.Data_Vector_Length]
    test_y = Drift_data_array[int(data_count * args.Train_Ratio):, -2]
    test_locy = Drift_data_array[int(data_count * args.Train_Ratio):, -1]
    y = np.hstack((train_y, test_y))
    n_classes = len(np.unique(y))
    if n_classes < args.Nc:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return train_x, train_y, train_locy, test_x, test_y, test_locy

if __name__ == '__main__':
    main()

