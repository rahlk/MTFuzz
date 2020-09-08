# MTFuzz: Fuzzing with a Multi-Task Neural Network

### Framework
![image](https://user-images.githubusercontent.com/57293631/80742593-a34d8100-8ae9-11ea-9f52-a1931d945a5c.png)

### Intro
MTFuzz is a novel neural network assisted fuzzer based on multi-task learning technique. It uses a NN to learn a compact embedding of input file for multiple fuzzing tasks (i.e., predicting different types of code coverage). The compact embedding is used to guide effective mutation by focusing on hot bytes. Our results show MTFuzz uncovers 11 previously unseen bugs and achieves an average of 2x more edge coverage compared with 5 state-of-the-art fuzzers on 10 real-world programs.

### Cite as
```
@article{she2020mtfuzz,
  title={MTFuzz: Fuzzing with a Multi-Task Neural Network},
  author={She, Dongdong and Krishna, Rahul and Yan, Lu and Jana, Suman and Ray, Baishakhi},
  journal={arXiv preprint arXiv:2005.12392},
  year={2020}
}
```

## Submission 

Published in ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE) 2020. 

ARXIV Link: [https://arxiv.org/pdf/2005.12392.pdf](https://arxiv.org/pdf/2005.12392.pdf)

## Cite As

```
@article{krishna2020whence,
  title={Whence to Learn? Transferring Knowledge in Configurable Systems using BEETLE},
  author={Krishna, Rahul and Nair, Vivek and Jamshidi, Pooyan and Menzies, Tim},
  journal={IEEE Transactions on Software Engineering},
  year={2020},
  publisher={IEEE}
}
```

## Authors

+ Dongdong She(a), Rahul Krishna(b), Lu Yan(c), Suman Jana(d), Baishakhi Ray(e)
  + (a) ds3619@columbia.edu, Columbia Univ., USA
  + (b) i.m.ralk@gmail.com, Columbia Univ., USA
  + (c) jiaodayanlu@sjtu.edu.cn, Shanghai Jiao Tong University Shanghai, China
  + (d) suman@cs.columbia.edu, Columbia Univ., USA
  + (e) rayb@cs.columbia.edu, Columbia Univ., USA

## Source code (Zenodo)

+ [Source Code](https://doi.org/10.5281/zenodo.3903818)


# 2. Setup (local Machine)

## 2.1 Install prerequisite
- Python 3.7 or more
- Install tensorflow-gpu 1.15 (Note that you need to install proper CUDA, CuDNN drivers before installing tensorflow-gpu. We recommend `conda` package manager for python. In our experience, it has done a good job installing all the dependencies).
- Install Keras 2.24
- Install LLVM 7.0.0 (we recommend building from the source)
- Install Clang 7

## 2.2 Build MTFuzz
```bash
    cd source
    ./build.sh  # build llvm coverage passes and CMP passes.
```

## 2.3 Run MTFuzz
Run MTFuzz on 10 tested programs reported in our paper. We will use program size as an example.

1. Enter size directory
```bash
    cd programs/size
```
2. Install some required libraries.
```bash
    sudo dpkg --add-architecture i386
    sudo apt-get update
    sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1
```
3. Set CPU scaling and core dump notification with root
```bash
    cd /sys/devices/system/cpu
    echo performance | tee cpu*/cpufreq/scaling_governor
    echo core >/proc/sys/kernel/core_pattern
```
4. Open two terminal sessions (let's call them _terminal-A_ and _terminal-B_). _Please Note: if you are on aws, (1) open a new terminal; (2) ssh into the aws container; and (3) cd to the current directory._

5. In terminal-A, start the MTNN module as follows (**Do not exit/close this terminal**)
```bash  
    python nn.py ./size 
```
5. In terminal-B, start fuzzing module (**Do not close/exit this terminal either**)
```bash
    # -l, file len is obtained by maximum file lens in the mtfuzz_in ( ls -lS mtfuzz_in|head )
    python ./mtfuzz_wrapper.py -i mtfuzz_in -o seeds -l 7402 ./size @@
```

_Note: The initial data processing will take around 5-10 minutes. If you see the following log in NN module terminal and fuzzing module terminal, then MTFuzz is running correctly. In fuzzing module terminal, the first red block shows the edge coverage of init seed corpus, then the following lines shows the current edge coverage discovered by MTFuzz. To compute the new edge coverage, users simply need to substrate init edge coverage from current edge coverage._

![image](https://github.com/Dongdongshe/fse20/blob/master/submissions/reusable/mtfuzz/nn_module.png?raw=true)
![image](https://github.com/Dongdongshe/fse20/blob/master/submissions/reusable/mtfuzz/fuzzing_module.png?raw=true)


# 3. Use MTFuzz on your own program (_Recommended for exending current method, this will take a lot more time_)
Here, we demonstrate how to set up MTFuzz on your own programs. Let's use expat, an XML parser, as an example. _Note: in the following instructions, we use some automation bash scripts like `build_expat.sh` or `./setup_mtfuzz_xmlwf.sh`. These are meant to ease the build process, please change them as you see fit for according to the specifics of your program._

1. Go to the programs directory inside [source](https://github.com/Dongdongshe/fse20/tree/master/submissions/reusable/mtfuzz/source) folder. If you are using our AWS, then use:
```bash
cd ~/mtfuzz/source/programs
```

2. Download and unzip expat source code and cd into expat's root directory:
```bash
    wget https://github.com/libexpat/libexpat/releases/download/R_2_2_9/expat-2.2.9.tar.bz2 
    tar -axvf expat-2.2.9.tar.bz2
    cd expat-2.2.9
```
2.  Instrument expat with MTFuzz llvm pass.
```bash
    cp ../build_expat.sh .
    ./build_expat.sh
```
3. Collect init training dataset for MTFuzz. Run afl-fuzz with a single input file on xmlwf for about an hour. To save time, we provide a collected dataset in xmlwf_afl_1hr directory. Set up MTFuzz for xmlwf.
```bash
    cd ..
    ./setup_mtfuzz_xmlwf.sh
```
4. Enter xmlwf directory and start nn module.
```bash
    cd xmlwf
    python ./nn.py ./xmlwf
```
5. Open another terminal and enter same directory and start fuzzing module
```bash
    # -l, file len is obtained by maximum file lens in the mtfuzz_in ( ls -lS mtfuzz_in|head )
    python ./mtfuzz_wrapper.py -i mtfuzz_in/ -o seeds/ -l 7961 ./xmlwf @@
```
You can find the following output log at the two terminals if MTFuzz runs correctly.
![image](https://github.com/Dongdongshe/fse20/blob/master/submissions/reusable/mtfuzz/xmlwf_nn.png?raw=true)
![image](https://github.com/Dongdongshe/fse20/blob/master/submissions/reusable/mtfuzz/xmlwf_fuzz.png?raw=true)


# License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

(BTW, it would be great to hear from you if you are using this material. But that is optional.)

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to http://unlicense.org

