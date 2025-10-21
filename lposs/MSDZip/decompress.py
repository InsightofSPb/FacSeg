import math
import shutil
import sys
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import compress_model
import arithmeticcoding_fast
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Source file.')
    parser.add_argument('output', type=str, help='Compressed file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
    parser.add_argument('--device', type=str, help='Torch device to use (overrides --gpu).')
    parser.add_argument('--tempdir', '-T', type=str, help='Temporary folder name.')
    parser.add_argument('--tempfile', type=str, help='TemFile Folder .')
    parser.add_argument('--prefix', '-p', type=str, help='Prefixes of files')
    parser.add_argument('--index', '-i', type=str, help='Index of files')
    parser.add_argument('--batchsize', '-b', type=int, default=512, help='Sample size in one batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-7, help='Weight decay.')
    parser.add_argument('--timesteps', type=int, default=16, help='The number of history symbols')
    parser.add_argument('--vocab_dim', type=int, default=16, help='The dimension of vocab.')
    parser.add_argument('--vocab_size', type=int, default=256, help='The size of vocab.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='The dimension of hidden layer.')
    parser.add_argument('--ffn_dim', type=int, default=4096, help='The dimension of ffn layer.')
    parser.add_argument('--layers', type=int, help='Num of layers')
    parser.add_argument('--seed', type=int, default=0, help='Random seeds.')
    parser.add_argument('--sp', action='store_true', help='Stepwise-parallel')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--load', action='store_true', help='Load the model')
    parser.add_argument('--ratio', type=float, default=0.05, help='Pretrain ratio.')
    parser.add_argument('--weights', type=str, help='Optional path to pretrained model weights (.pth).')
    parser.add_argument('--mode', choices=['adaptive', 'static'], default='adaptive',
                        help='Compression mode used during encoding. Must match the encoder configuration.')
    args = parser.parse_args(argv)
    return args


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        device = torch.device(args.device)
        if device.type == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        return device
    if args.gpu.lower() == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        return torch.device('cuda')
    logging.warning('CUDA was requested but is not available. Falling back to CPU.')
    return torch.device('cpu')

def decompress(args, temp_file, info, last, device: torch.device):
    bs, ts = args.batchsize, args.timesteps
    len_series, vocab_size = info[args.sub_prefix], args.vocab_size


    iter_num = (len_series - ts) // bs      # 10439

    # print(iter_num - ts)
    series_2d = np.zeros((bs, iter_num), dtype=np.uint8).astype('int')

    f = [open(temp_file + '.' + str(i), 'rb') for i in range(bs)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

    prob = np.ones(vocab_size) / vocab_size
    cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    # Decode first K symbols in each stream with uniform probabilities
    for i in range(bs):
        for j in range(min(ts, iter_num)):
            series_2d[i, j] = dec[i].read(cumul, vocab_size)

    cumul_batch = np.zeros((bs, vocab_size + 1), dtype=np.uint64)
    model = compress_model.MixedModel(batchsize=args.batchsize, layers=args.layers, hidden_dim=args.hidden_dim,
                                      ffn_dim=args.ffn_dim, vocab_size=vocab_size, vocab_dim=args.vocab_dim,
                                      timesteps=ts).to(device)   #没有用到vocab_dim

    if args.weights:
        logging.info('Loading pretrained weights from %s', args.weights)
        state, source_key, stripped = load_checkpoint_state(args.weights, device)
        if source_key != 'root':
            logging.info("Extracted state_dict from checkpoint key '%s'", source_key)
        if stripped:
            logging.info("Removed 'module.' prefix from checkpoint parameters")
        try:
            model.load_state_dict(state)
        except RuntimeError:
            logging.exception('Failed to load weights from %s', args.weights)
            raise
    elif args.load:
        logging.info('Loading Model!')
        model.load_state_dict(torch.load(args.prefix + '_model/{}.{}.pth'.format(args.prefix, int(args.index)-1), map_location=device, weights_only=True))

    adaptive = args.mode == 'adaptive'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) if adaptive else None

    if iter_num - ts > 0:
        for train_index in range(iter_num - ts):
            context = torch.from_numpy(series_2d[:, train_index:train_index + ts]).to(device).long()
            with torch.set_grad_enabled(adaptive):
                if adaptive:
                    model.train()
                else:
                    model.eval()
                logits = model.forward(context)
            prob = logits[:, -1, :]
            prob = F.softmax(prob, dim=1).detach().cpu().numpy()
            cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            # Decode with Arithmetic Encoder
            for i in range(bs):
                series_2d[i, train_index + ts] = dec[i].read(cumul_batch[i, :], vocab_size)

            if adaptive:
                target = torch.from_numpy(series_2d[:, train_index + ts]).to(device).long()
                loss = torch.nn.functional.cross_entropy(logits[:, -1, :], target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if adaptive and train_index == int((iter_num - ts) * args.ratio) and args.save:
                logging.info('Saving Model!')
                torch.save(model.state_dict(), args.prefix + '_model/{}.{}.pth'.format(args.prefix, args.index))

    fout = open(args.output, 'wb')
    fout.write(bytearray(series_2d.reshape(-1).tolist()))

    for i in range(bs):
        bitin[i].close()
        f[i].close()

    if last:
        series = np.zeros(last, dtype=np.uint8).astype('int')
        f = open(temp_file + '.last', 'rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(last):
            series[j] = dec.read(cumul, vocab_size)
        fout.write(bytearray(series.tolist()))

        # Avoid the bug where the program waits indefinitely due to the absence of an error-reporting model.
        if args.save:
            with open(args.prefix + '_model/{}.{}.pth'.format(args.prefix, args.index), 'w') as f_model:
                f_model.write('')
            f_model.close()


        bitin.close()
        f.close()
    return

def main(args):
    t1 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    device = resolve_device(args)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    logging.info('Using device %s', device)

    if args.layers is None:
        args.layers = int(math.log2(args.timesteps) + 1)

    if args.prefix is None:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]

    if args.sp:
        args.sub_prefix = args.prefix + '.' + args.index
    else:
        args.sub_prefix = args.prefix

    if not args.tempdir:
        args.tempdir = "{}_bs{}_ts{}_v{}_h{}_f{}_l{}".format(args.sub_prefix, args.batchsize, args.timesteps, args.vocab_dim, args.hidden_dim, args.ffn_dim, args.layers)


    # args.timesteps = args.timesteps * (args.hidden_dim // args.vocab_dim)
    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    temp_file = args.tempdir + '/compressed_temp_file'
    params = eval(open(args.prefix + '.params', 'r').read())

    f = open(args.input, 'rb')
    for i in range(args.batchsize):
        f_out = open(temp_file + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        # print(byte_str)
        f_out.write(byte_str)
        f_out.close()

    f_out = open(temp_file + '.last', 'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    if (params[args.sub_prefix] - args.timesteps) % args.batchsize == 0:
        decompress(args, temp_file, params, 0, device)
    else:
        last_length = (params[args.sub_prefix] - args.timesteps) % args.batchsize + args.timesteps
        decompress(args, temp_file, params, last_length, device)
    # remove temp files
    shutil.rmtree(args.tempdir)
    t2 = time.time()
    logging.info('{} has been decompressed, in {} secs.'.format(args.sub_prefix, int(t2 - t1)))
    # print('Peak GPU memory usage: {} KBs'.format(torch.cuda.max_memory_allocated()//1024))

def setupLogging(debug=False):
    logLevel = logging.DEBUG if debug else logging.INFO
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stderr, level=logLevel, format=logFormat)
    logging.info("Running %s" % " ".join(sys.argv))

def run(argv):
    setupLogging()
    args = parseArgs(argv)
    starttime = time.time()
    main(args)

if __name__ == '__main__':
    run(sys.argv[1:])