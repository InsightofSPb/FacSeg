<div align="center">
<h1>[WWW '25] MSDZip: Universal Lossless Compression for Multi-source Data via Stepwise-parallel and Learning-based Prediction</h1>
<h3>Huidong Ma, Hui Sun, Liping Yi, Yanfeng Ding, Xiaoguang Liu, Gang Wang</h3>
</div>

# Running
## Regular
```
# Compression
python compress.py <file> <file>.mz --prefix <prefix>
# Decompression
python decompress.py <file>.mz <file>.mz.out --prefix <prefix>
```
For example
```
python compress.py enwik6 enwik6.mz --prefix enwik6
python decompress.py enwik6.mz enwik6.mz.out --prefix enwik6
```

### Advanced options

The compressor and decompressor now expose several flags that make it easier
to run controlled experiments and analyse the output quality:

* `--mode adaptive|static` toggles between the original adaptive training
  behaviour (`adaptive`, default) and a frozen evaluation-only mode (`static`).
  Make sure that the encoder and decoder use the same value.
* `--weights <path>` loads a checkpoint containing model weights before
  compression/decompression starts. This can be used to compare the baseline
  MSDZip model, a fine-tuned variant trained with `lposs_train`, or any other
  checkpoint.
* `--device` lets you force a particular PyTorch device (for example `cpu`).
* `--metrics-path <path>` stores a per-symbol bitrate trace (NumPy `.npz`)
  which can be inspected with `analyze_metrics.py`. Combine this with
  `--metrics-topk` to log the worst-performing indices directly to stdout.

Use the helper script below to visualise the metric trace and highlight the
regions that are hard to compress:

```
python analyze_metrics.py --metrics <trace.npz> --resolution 512x512 \
    --heatmap heatmap.pgm --mask hot-spots.pgm --threshold 5.0 --topk 20
```
The script will emit basic statistics, optionally save a heatmap or binary
mask (using the PGM format so no external dependencies are required), and
report the coordinates of the locations with the highest bitrate.

## Stepwise-parallel
```
# Compression
bash sp-compress.sh <file> <file>.mz <prefix> <parallel>
# Decompression
bash sp-decompress.sh <file>.mz <file>.mz.out <prefix> <parallel>
```
For example
```
bash sp-compress.sh enwik6 enwik6.mz enwik6 2
bash sp-decompress.sh enwik6.mz enwik6.mz.out enwik6 2
```

# Dataset
| ID  | Name           | Type          | Size (Byte)   | Link                                                                                   |
|:---:|:--------------:|:-------------:|:-------------:|:--------------------------------------------------------------------------------------:|
| D1  | Enwik8         | text          | 100000000     | https://mattmahoney.net/dc/enwik8.zip                                                  |
| D2  | Text8          | text          | 100000000     | https://mattmahoney.net/dc/text8.zip                                                   |
| D3  | Enwik9         | text          | 1000000000    | https://mattmahoney.net/dc/enwik9.zip                                                  |
| D4  | Book           | text          | 1000000000    | https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2  |
| D5  | Silesia        | heterogeneous | 211938580     | https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip                                    |
| D6  | Backup         | heterogeneous | 1000000000    | https://drive.google.com/file/d/18qvfbeeOwD1Fejq9XtgAJwYoXjSV8UaC/view?usp=sharing     |
| D7  | CLIC           | image         | 243158876     | https://www.compression.cc/tasks/                                                      |
| D8  | ImageTest      | image         | 470611702     | http://imagecompression.info/test_images/rgb8bit.zip                                   |
| D9  | GoogleSpeech   | audio         | 327759206     | http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz                       |
| D10 | LJSpeech       | audio         | 293847664     | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2                             |
| D11 | DNACorpus      | genome        | 685597124     | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip                                      |
| D12 | GenoSeq        | genome        | 1926041160    | https://www.ncbi.nlm.nih.gov/sra/ERR7091247                                            |

# Citation
If you are interested in our work, we hope you might consider starring our repository and citing our [paper](https://dl.acm.org/doi/10.1145/3696410.3714655):
```
@inproceedings{ma2025msdzip,
  title={MSDZip: Universal Lossless Compression for Multi-source Data via Stepwise-parallel and Learning-based Prediction},
  author={Ma, Huidong and Sun, Hui and Yi, Liping and Ding, Yanfeng and Liu, Xiaoguang and Wang, Gang},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={3543--3551},
  year={2025}
}
```

# Acknowledgment
The code is based on [PAC](https://github.com/mynotwo/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework) and [Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding). Thanks for these great works.

# Contact
Email: mahd@nbjl.nankai.edu.cn  
Nankai-Baidu Joint Laboratory (NBJL)
