import numpy as np
import torch
import json
import torch.optim as optim
from prototypical_batch_sampler import PrototypicalBatchSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import data_handler as dh
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset,DataLoader

class Strategy:
    def __init__(self, X, Y, Z, original_data, net, args):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.original_data = original_data
        self.model = net
        self.args = args

    def init_lr_scheduler(self, optim):
        '''
        Initialize the learning rate scheduler
        '''
        return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=self.args.lr_scheduler_gamma,
                                               step_size=self.args.lr_scheduler_step)

    def init_sampler(self, labels):
        classes_per_it = self.args.Nc
        num_samples = self.args.Ns + self.args.Nq

        return PrototypicalBatchSampler(labels=labels,
                                        classes_per_it=classes_per_it,
                                        num_samples=num_samples,
                                        iterations=self.args.iterations)
    def init_dataloader(self, supple_data, mark_x, mark_y, mark_l):
        train_x, train_y, train_locy, test_x, test_y, test_locy = \
            supple_data[0], supple_data[1], supple_data[2], supple_data[3], supple_data[4], supple_data[5]
        train_x = np.concatenate([train_x, mark_x], axis=0)
        train_y = np.concatenate([train_y, mark_y], axis=0)
        train_locy = np.concatenate([train_locy, mark_l], axis=0)
        arr = np.arange(train_x.shape[0])
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        train_locy = train_locy[arr]

        sampler = self.init_sampler(train_y)
        # TODO train DataLoader
        Train_DS = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y),
                                 torch.unsqueeze(torch.FloatTensor(train_locy), 1))
        train_dataloader = DataLoader(Train_DS, batch_sampler=sampler)
        # TODO test DataLoader
        sampler = self.init_sampler(test_y)
        Test_DS = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y),
                                torch.unsqueeze(torch.FloatTensor(test_locy), 1))
        test_dataloader = DataLoader(Test_DS, batch_sampler=sampler)
        return train_dataloader, test_dataloader

    def train(self, supple_data):
        n_epoch = self.args.epochs
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        # TODO train DataLoader
        train_dataloader, test_dataloader = self.init_dataloader(supple_data, self.X, self.Y, self.Z)
        lr_scheduler = self.init_lr_scheduler(optimizer)
        train_loss = []
        train_class_acc = []
        train_loc_acc = []
        test_loss = []
        test_class_acc = []
        test_loc_acc = []
        centroid_matrix = torch.Tensor()
        # Training loop
        for i in range(n_epoch):
            self.model.train()
            for batch_idx, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                datax, datay, locy = data
                loss, class_acc, loc_acc, centroid_matrix = self.model(datax, datay, locy)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_class_acc.append(class_acc.item())
                train_loc_acc.append(loc_acc.item())
                centroid_matrix = centroid_matrix

            avg_loss = np.mean(train_loss)
            avg_class_acc = np.mean(train_class_acc)
            avg_loc_acc = np.mean(train_loc_acc)
            print('{} episode,Avg Train Loss: {}, Avg Train Class Acc: {}, Avg Train loc Acc: {}'.format(
                i, avg_loss, avg_class_acc, avg_loc_acc))

            lr_scheduler.step()

        PATH = self.args.pretrain_model_path + '/{name}_model_embeding.pkl'.format(name=self.args.FAN)
        torch.save(self.model.state_dict(), PATH)
        # save centroid matrix
        torch.save(centroid_matrix,
                   self.args.pretrain_model_path + '/{name}_centroid_matrix.pt'.format(name=self.args.FAN))
        # save loss
        with open(self.args.pretrain_model_path + '/{name}_loss.json'.format(name=self.args.FAN), "w") as f:
            json.dump(
                {"train_loss": np.array(train_loss[len(train_loss) - 5:]).sum() / len(train_loss[len(train_loss) - 5:]),
                 "train_class_acc": np.array(train_class_acc[len(train_class_acc) - 5:]).sum() / len(
                     train_class_acc[len(train_class_acc) - 5:]),
                 "train_loc_acc": np.array(train_loc_acc[len(train_loc_acc) - 5:]).sum() / len(
                     train_loc_acc[len(train_loc_acc) - 5:])}, f)
        # Test loop
        for batch_idx, data in enumerate(test_dataloader):
            datax, datay, locy = data
            loss, class_acc, loc_acc, centroid_matrix = self.model(datax, datay, locy)
            test_loss.append(loss.item())
            test_class_acc.append(class_acc.item())
            test_loc_acc.append(loc_acc.item())
        avg_class_acc = np.mean(test_class_acc)
        avg_loc_acc = np.mean(test_loc_acc)
        # save result
        with open(self.args.pretrain_model_path + '/{name}_result.json'.format(name=self.args.FAN), "w") as f:
            json.dump({"avg_class_acc": avg_class_acc, "avg_loc_acc": avg_loc_acc}, f)

        print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
            avg_class_acc, avg_loc_acc))

    def predict(self, BASE_PATH):
        # detection
        m = 6
        the_k = 15
        compare_window_len = 765
        stream = dh.DriftDataset('./datasets/', self.args.Dataset).np_data
        stream_X = stream[:, :-1]
        stream_Y = stream[:, -1]
        rid, idx, correct_sample, sample_num = 0, 0, 0, 0
        Y_pred = []
        accuracyList = []
        learning_model = None
        inital_model = True
        while True:
            if inital_model:
                learning_model = GaussianNB()
                learning_model.fit(stream_X[idx: idx + m], stream_Y[idx: idx + m])
                for _ in range(m):
                    Y_pred.append(0)
                    idx = idx + 1
                inital_model = False

            sample_num += 1
            X = stream_X[idx: idx + 1]
            Y = stream_Y[idx: idx + 1]
            Y_hat = learning_model.predict(X)
            Y_pred.append(Y_hat[0])
            if Y_hat[0] - Y[0] == 0:
                correct_sample += 1

            if idx + 1 == stream_X.shape[0]:
                break
            learning_model.fit(stream_X[rid: idx + 1], stream_Y[rid: idx + 1])
            if sample_num % compare_window_len != 0:
                if sample_num % the_k == 0:
                    accuracyList.append(correct_sample / sample_num)
                idx += 1
            else:
                error_feature = []
                accuracyList.append(correct_sample / sample_num)
                for i in range(len(accuracyList) - 1):
                    j = i + 1
                    if accuracyList[i] == 0.0:
                        error_feature.append(0.0)
                    else:
                        error_feature.append((accuracyList[j] - accuracyList[i]) / accuracyList[i])

                # TODO Type-LDA start
                pre_type, pre_type_score, pre_loc = self.model(torch.tensor([error_feature]), BASE_PATH)
                # pre_type, pre_loc = 2, 32
                if pre_type.item() < 3:
                    point = int(pre_loc.item()) + 1
                    if point <= 0:
                        idx = idx - 1 * the_k
                    else:
                        idx = idx - point * the_k
                    rid = idx
                    Y_pred = Y_pred[:idx]
                    inital_model = True
                else:
                    idx += 1

                correct_sample = 0
                sample_num = 0
                accuracyList = []

        acc_slide_chunk = accuracy_score(stream_Y, Y_pred)
        print('accuracy：{}'.format(acc_slide_chunk))

    def GaussianNB_Baseline(self):
        # baseline
        stream = dh.DriftDataset('./datasets/', self.args.Dataset).np_data
        stream_X = stream[:, :-1]
        stream_Y = stream[:, -1]
        init_train_size = 200
        first_batch = True
        num_batch = int(stream_X.shape[0] / init_train_size)
        stream_spliter = KFold(n_splits=num_batch, random_state=None, shuffle=False)
        Y_pred = [0, 0, 0, 0, 0]

        for _, batch_idx in stream_spliter.split(stream_X):

            batch_X = stream_X[batch_idx]
            batch_Y = stream_Y[batch_idx]

            if first_batch:
                learning_model = GaussianNB()
                learning_model.fit(batch_X[:5], batch_Y[:5])
                for i in range(5, batch_X.shape[0]):
                    Y_hat = learning_model.predict(batch_X[i:i + 1])
                    Y_pred.append(Y_hat[0])
                    learning_model.fit(batch_X[:i + 1], batch_Y[:i + 1])
                first_batch = False
            else:
                batch_Y_hat = learning_model.predict(batch_X)
                Y_pred = np.append(Y_pred, batch_Y_hat)
                learning_model.fit(batch_X, batch_Y)

        acc_slide_chunk = accuracy_score(stream_Y, Y_pred)
        print("贝叶斯baseline:{}".format(acc_slide_chunk))
    def predict_prob(self, X, BASE_PATH):
        # TODO test DataLoader
        Test_DS = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        loader_test = DataLoader(Test_DS, batch_size=3, shuffle=False)
        self.model.eval()
        types = torch.Tensor()
        probs = torch.Tensor()
        locs = torch.Tensor()
        with torch.no_grad():
            for idx, (x,_) in enumerate(loader_test):
                pre_type_y, pre_type_score, pre_loc_y = self.model(x, BASE_PATH)
                if idx > 0:
                    types = torch.concat([types, pre_type_y], dim=0)
                    probs = torch.concat([probs, pre_type_score], dim=0)
                    locs = torch.concat([locs, pre_loc_y], dim=0)
                else:
                    types = pre_type_y
                    probs = pre_type_score
                    locs = pre_loc_y

        return types, probs, locs





