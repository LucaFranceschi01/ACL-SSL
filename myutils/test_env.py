import argparse
import torch
import os

def main(*args):
    print(args)
    print(f'{USE_CUDA=}')
    print(f'{NUM_GPUS=}')
    print(f'{USE_DDP=}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Use model config file name')
    parser.add_argument('--model_path', type=str, default='', help='Use model save path')
    parser.add_argument('--train_config', type=str, default='', help='Use train config file name')
    parser.add_argument('--exp_name', type=str, default='', help='postfix for save experiment')
    parser.add_argument('--save_path', type=str, default='', help='Save path for model and results')
    parser.add_argument('--vggss_path', type=str, default='', help='VGGSS dataset directory')
    parser.add_argument('--flickr_path', type=str, default='', help='Flickr dataset directory')
    parser.add_argument('--avs_path', type=str, default='', help='AVSBench dataset directory')
    parser.add_argument('--vggsound_path', type=str, default='', help='VGGSound dataset directory')
    parser.add_argument('--local_rank', type=str, default='', help='Rank for distributed train')
    parser.add_argument('--san', action='store_true', help='Silence and noise implementation during training')

    args = parser.parse_args()

    data_path = {'vggss': args.vggss_path,
                 'flickr': args.flickr_path,
                 'avs': args.avs_path,
                 'vggsound': args.vggsound_path}

    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    NUM_GPUS = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    USE_DDP = True if NUM_GPUS > 1 else False

    rank = 0 if not USE_DDP else None

    print(f'{torch.version.cuda=}')
    print(f'{torch.__version__=}')

    # Run example
    main(args.model_name, args.model_path, args.exp_name, args.train_config, data_path, args.save_path, args.san)