import argparse

class myParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='0, 1, 2, 3, 4, 5')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--model', type=str, default='fedavg')
        self.parser.add_argument('--dataset', type=str, default='cifar10')

        self.parser.add_argument('--n-workers', type=int, default=10)
        self.parser.add_argument('--n-clients', type=int, default=10)
        self.parser.add_argument('--frac', type=float, default=1.0)
        self.parser.add_argument('--n-rnds', type=int, default=10)
        self.parser.add_argument('--n-eps', type=int, default=2)

        # These two is for deciding how to tune the similarity matrix
        self.parser.add_argument('--agg-norm', type=str, default='exp', choices=['cosine', 'exp'])
        self.parser.add_argument('--norm-scale', type=float, default=3)

        self.parser.add_argument('--base-path', type=str, default='../')

        self.parser.add_argument('--mode', type=str, default='random_iid', choices=['ramdom_iid', 'pathological_pair', 'cluster_dirichlet_v1'])

        # Hyperparameter for FedTS
        self.parser.add_argument('--ready-cosine', type=float, default=0.99)
        self.parser.add_argument('--epoch-limit', type=int, default=200) # Do not implement succesfully yet
        self.parser.add_argument('--matching-rounds', type=int, default=20)

        # Hyperparameter for custom_sim
        self.parser.add_argument("--custom-epochs-list", nargs="+", type=int, default=[10, 20, 30], help="A list of integers denoting when to do matching")

    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit(f'Unknown argument: {unparsed}')
        return args # 這會是一個Namespace對象，用args.gpu, args.seed來訪問 !