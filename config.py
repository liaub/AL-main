import argparse
import os
args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0,help="gpu")
args.add_argument('--Data-Vector-Length', type=int, default=50,
                  help='Setting original_data Dimensions')
args.add_argument('--centord-Vector-Length', type=int, default=30,
                  help='Setting centord Vector Dimensions')
args.add_argument('--DATA-FILE', type=str, default='./datasets/1_SyntData_SyntDrift/pretrain_data/SEAa0/',
                  help='Select original_data set')
args.add_argument('--Dataset', type=str, default="RBFi")
args.add_argument('--DRIFT-FILE', type=str, default='./datasets/drift-50-15-4800/',
                  help='Additional training original_data to solve the problem of less fine-tuning original_data')
args.add_argument('--BASE-PATH', type=str, default='./models/drift-50-15',
                  help='Basic model')
args.add_argument('--Train-Ratio', type=float, default=0.2,
                  help='Training set ratio')
args.add_argument('--DATA-SAMPLE-NUM', type=int, default=4800,
                  help='Number of original_data sets sampled')
args.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs for training (default: 20)')
args.add_argument('--RNN', type=str, default="RNN")
args.add_argument('--FAN', type=str, default="FAN")
args.add_argument('--FCN', type=str, default="FCN")
args.add_argument('--FQN', type=str, default="FQN")
args.add_argument('--FNN', type=str, default="FNN")
args.add_argument('--CNN', type=str, default="CNN")
args.add_argument('--PNN', type=str, default="PNN")
args.add_argument('--frame-size', type=int, default=200)
args.add_argument('--train-original_data', type=str, default="train")
args.add_argument('--test-original_data', type=str, default="test")

args.add_argument('--num-episode', type=int, default=600)
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--lr-scheduler-gamma', type=float, default=0.9)
args.add_argument('--lr-scheduler-step', type=int, default=30)
args.add_argument('--alpha', type=float, default=0.7)
args.add_argument('--query-num', type=int, default=3)





args.add_argument('--Ns', type=int,
                  help='number of samples per class to use as support for training, default=5',
                  default=5)
args.add_argument('--Nc', type=int,
                  help='number of random classes per episode for training, default=4',
                  default=4)
args.add_argument('--Nq', type=int,
                  help='number of samples per class to use as query for training, default=5',
                  default=5)
args.add_argument('--iterations', type=int,
                  help='number of episodes per epoch, default=100',
                  default=200)

args.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
args = args.parse_args()

if not os.path.exists(args.DATA_FILE + 'mark'):
    os.mkdir(args.DATA_FILE + 'mark')

if not os.path.exists('./models/' + args.DATA_FILE.split('/')[-2]):
    os.mkdir('./models/' + args.DATA_FILE.split('/')[-2])
args.pretrain_model_path = './models/' + args.DATA_FILE.split('/')[-2]

