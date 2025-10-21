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

from typing import Optional

import compress_model
import arithmeticcoding_fast
from utils import *
from metrics import CompressionMetricsRecorder
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Source file.')
    parser.add_argument('output', type=str, help='Compressed file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
    parser.add_argument('--device', type=str, help='Torch device to use (overrides --gpu).')
    parser.add_argument('--tempdir', '-T', type=str, help='Temporary folder name.')
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
                        help='Compression mode. "adaptive" keeps training the model during compression, '
                             '"static" only evaluates without updating weights.')
    parser.add_argument('--metrics-path', type=str,
                        help='If provided, store per-symbol bitrate metrics at this path (npz format).')
    parser.add_argument('--metrics-topk', type=int, default=0,
                        help='When saving metrics, print the hardest K positions to stdout.')
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


def compress(args, temp_file, series, train_data, final, device: torch.device,
             metrics: Optional[CompressionMetricsRecorder] = None, series_offset: int = 0,
             checkpoint: Optional[dict] = None):
    bs, ts = args.batchsize, args.timesteps
    f = [open(temp_file + '.' + str(i), 'wb') for i in range(bs)]  # 创建batchsize个空文件

    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]  # 同DZip
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

    prob = np.ones(args.vocab_size) / args.vocab_size  # 初始化一个概率       vocab_size = 256  len(prob)=256
    cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)  # 累加        266
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    stream_total = len(train_data) // bs  # 每个batch流包含的symbol数量
    ind = np.array(range(bs)) * stream_total  # 批index
    # train_data = reorder_data(train_data, bs, iter_num)
    iter_num = stream_total - ts
    for i in range(bs):
        for j in range(ts):
            symbol_index = series_offset + ind[i] + j
            enc[i].write(cumul, series[ind[i] + j])
            if metrics is not None:
                metrics.record_uniform([symbol_index])
    cumul_batch = np.zeros((bs, args.vocab_size + 1), dtype=np.uint64)  # [128, 256+1]  # 原来是vocab_size
    model = compress_model.MixedModel(batchsize=args.batchsize, layers=args.layers, hidden_dim=args.hidden_dim,
                                      ffn_dim=args.ffn_dim, vocab_size=args.vocab_size,
                                      vocab_dim=args.vocab_dim, timesteps=ts).to(device)  # 没有用到vocab_dim

    state = None
    source_key = None
    stripped = False
    if checkpoint is not None:
        state = checkpoint.get('state_dict')
        source_key = checkpoint.get('source_key')
        stripped = checkpoint.get('stripped', False)
    elif args.weights:
        logging.info('Loading pretrained weights from %s', args.weights)
        state, source_key, stripped, _ = load_checkpoint_state(args.weights, device)
    if state is not None:
        if args.weights:
            logging.info('Applying checkpoint weights from %s', args.weights)
        if source_key and source_key != 'root':
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

    if iter_num > 0:
        for train_index in range(iter_num):
            if train_index % 10 == 0:
                print(f"{train_index + 1}/{iter_num}")
            row_indices = ind.copy()
            train_batch = train_data[row_indices, :]
            y = train_batch[:, -1]
            context = torch.from_numpy(train_batch[:, :-1]).to(device).long()
            with torch.set_grad_enabled(adaptive):
                if adaptive:
                    model.train()
                else:
                    model.eval()
                logits = model.forward(context)
            logits_last = logits[:, -1, :]
            if adaptive:
                target = torch.from_numpy(y).to(device).long()
                loss = torch.nn.functional.cross_entropy(logits_last, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prob = logits_last
            prob = F.softmax(prob, dim=1).detach().cpu().numpy()
            cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            for i in range(bs):
                enc[i].write(cumul_batch[i, :], y[i])
            if metrics is not None:
                symbol_positions = series_offset + row_indices + ts
                sample_prob = prob[np.arange(bs), y]
                metrics.record_probabilities(symbol_positions, sample_prob)
            ind += 1

            if adaptive and train_index == int(iter_num * args.ratio) and args.save:
                logging.info('Saving Model!')
                torch.save(model.state_dict(), args.prefix + '_model/{}.{}.pth'.format(args.prefix, args.index))

    for i in range(bs):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if final is not None:
        f = open(temp_file + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(args.vocab_size) / args.vocab_size
        cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(len(final)):
            enc.write(cumul, final[j])
        if metrics is not None:
            start_index = series_offset + stream_total * bs
            positions = np.arange(start_index, start_index + len(final))
            metrics.record_uniform(positions)

        # Avoid the bug where the program waits indefinitely due to the absence of an error-reporting model.
        if args.save:
            with open(args.prefix + '_model/{}.{}.pth'.format(args.prefix, args.index), 'w') as f_model:
                f_model.write('')
            f_model.close()

        enc.finish()
        bitout.close()
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

    checkpoint_bundle = prepare_state_from_weights(args, device, logger=logging.getLogger(__name__))

    if args.layers is None:
        args.layers = int(math.log2(args.timesteps) + 1)

    if not args.prefix:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]

    if args.sp:
        args.sub_prefix = args.prefix + '.' + args.index
    else:
        args.sub_prefix = args.prefix

    if not args.tempdir:
        args.tempdir = "{}_bs{}_ts{}_v{}_h{}_f{}_l{}".format(args.sub_prefix, args.batchsize, args.timesteps, args.vocab_dim, args.hidden_dim, args.ffn_dim, args.layers)

    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    temp_file = args.tempdir + '/compressed_temp_file'

    # args.timesteps = args.timesteps * (args.hidden_dim // args.vocab_dim)
    # Read input source file, and record key information
    with open(args.input, 'rb') as f:  # 一次一个byte = 8bit
        series = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()

    if not args.sp:
        params = dict()
        params[args.sub_prefix] = len(series)
        with open(args.prefix + '.params', 'w') as f:
            f.write(str(params))
        f.close()

    # Generating training data
    train_data = strided_app(series, args.timesteps + 1, 1)

    # Stat vocab freq
    total_num = len(train_data)  # sentence的个数
    metrics_recorder = None
    if args.metrics_path:
        metrics_recorder = CompressionMetricsRecorder(len(series), args.vocab_size)

    if total_num % args.batchsize == 0:  # 正好够整数个bs
        compress(args, temp_file, series, train_data, None, device, metrics_recorder, series_offset=0, checkpoint=checkpoint_bundle)
    else:  # 不够整数个batchsize
        ini_num = total_num // args.batchsize * args.batchsize  # 只压缩整数批的数据，整数个批里面有l+timesteps个元素
        # print(1, ini_num+args.timesteps)
        compress(args, temp_file, series[:ini_num + args.timesteps], train_data[:ini_num],
                 series[ini_num:], device, metrics_recorder, series_offset=0, checkpoint=checkpoint_bundle)

    # Combined compressed results
    f = open(args.output, 'wb')
    for i in range(args.batchsize):
        f_in = open(temp_file + '.' + str(i), 'rb')
        byte_str = f_in.read()  # 写入的二进制文件，固定写入，记住就行了
        byte_str_len = len(byte_str)  # 长度
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    if total_num % args.batchsize != 0:
        f_in = open(temp_file + '.last', 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f.close()

    total = 0
    for ff in os.listdir(args.tempdir):
        total += os.path.getsize(args.tempdir + '/' + ff)

    # Remove temp file
    shutil.rmtree(args.tempdir)
    t2 = time.time()
    f1_size, f2_size = os.stat(args.input).st_size, os.stat(args.output).st_size
    logging.info('{} has been compressed, with a compression ratio of {} bits/base, in {} secs.'.format(args.sub_prefix, round(f2_size / f1_size * 8, 3), int(t2 - t1)))

    if metrics_recorder and args.metrics_path:
        metadata = {
            'input': args.input,
            'mode': args.mode,
            'weights': args.weights,
            'batchsize': args.batchsize,
            'timesteps': args.timesteps,
            'vocab_size': args.vocab_size,
            'adaptive': args.mode == 'adaptive',
        }
        metrics_recorder.save(args.metrics_path, metadata)
        logging.info('Saved bitrate metrics to %s', args.metrics_path)
        if args.metrics_topk and args.metrics_topk > 0:
            hardest = metrics_recorder.topk(args.metrics_topk)
            if hardest:
                logging.info('Top-%d hardest symbols (index, bits): %s', args.metrics_topk, hardest)
            else:
                logging.info('No metrics were recorded; check input size or configuration.')

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