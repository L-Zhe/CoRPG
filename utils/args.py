import  argparse
from    math import inf


def get_train_parser():

    parser = argparse.ArgumentParser()
    parser = get_model_config(parser)
    parser = get_train_config(parser)
    parser = get_checkpoint_config(parser)
    parser = get_data_config(parser)

    return parser.parse_args()


def get_model_config(parser):

    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_layer_decoder', type=int, default=6)
    parser.add_argument('--num_layer_encoder', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--dropout_embed', type=float, default=0.1)
    parser.add_argument('--dropout_sublayer', type=float, default=0.1)
    parser.add_argument('--normalize_before', action='store_true')
    parser.add_argument('--num_sample', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_learning_rate', type=float, default=0)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clipper', type=float, default=5)
    parser.add_argument('--latent_dim', type=int, default=128)

    return parser


def get_train_config(parser):

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+')
    parser.add_argument('--batch_print_info', type=int, default=200)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=500)

    return parser


def get_data_config(parser):

    parser.add_argument('--clip_length', type=int, default=inf)
    parser.add_argument('--discard_invalid_data', action='store_true')
    parser.add_argument('--vocab', type=str, default='./')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--gram_penalty', type=float, nargs='+', default=[4, 2, 0, 0])
    parser.add_argument('--graph_eps', type=float, default=0.5)
    return parser


def get_checkpoint_config(parser):

    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--restore_file', type=str, default=None)
    parser.add_argument('--checkpoint_num', type=int, default=inf)
    parser.add_argument('--checkpoint_step', type=int, default=1000)

    return parser


def get_generate_config():

    parser = argparse.ArgumentParser()
    parser = get_model_config(parser)
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+')
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--raw_file', type=str, default=None)
    parser.add_argument('--ref_file', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=inf)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--decode_method', type=str, default='greedy')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--graph_eps', type=float, default=0.5)

    return parser.parse_args()


def get_preprocess_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--graph', type=str)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--save_file', type=str)

    return parser.parse_args()
