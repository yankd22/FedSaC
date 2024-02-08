import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid_2', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="/home/datadir/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 20, help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--concen_loss', type=str, default='uniform_norm', choices=['norm', 'uniform_norm'], help='How to measure the modle difference')
    parser.add_argument('--weight_norm', type=str, default='relu', choices=['sum', 'softmax', 'abs', 'relu', 'sigmoid'], help='How to measure the modle difference')
    parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
    parser.add_argument('--complementary_metric', type=str, default='PA', help='How to measure the model complementary')
    parser.add_argument('--matrix_alpha', type=float, default=1.2, help='Hyper-parameter for matrix alpha')
    parser.add_argument('--lam', type=float, default=0.01, help="Hyper-parameter in the objective")
    parser.add_argument('--k_principal', type=float, default=3, help='the dimension of the principal component')
    parser.add_argument('--matrix_beta', type=float, default=1.2, help='Hyper-parameter for matrix beta')
    parser.add_argument('--hidden_dim', type=int, default=84, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--alpha_bound', type=float, default=1, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--target_dir', type=str, default="/home", help='Hyper-parameter to avoid concentration')
    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')
    args = parser.parse_args()
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["epochs"] = args.epochs
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    cfg['client_num'] = args.n_parties
    cfg['model_name'] = args.model
    cfg['self_wight'] = 'loss'
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg