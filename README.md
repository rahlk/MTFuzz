# MTFuzz: Fuzzing with a Multi-Task Neural Network
### Paper link
https://dl.acm.org/doi/pdf/10.1145/3368089.3409723
### Framework
![image](https://user-images.githubusercontent.com/57293631/80742593-a34d8100-8ae9-11ea-9f52-a1931d945a5c.png)

### Abstract
Fuzzing is a widely used technique for detecting software bugs and vulnerabilities. Most popular fuzzers generate new inputs using an evolutionary search to maximize code coverage. Essentially, these fuzzers start with a set of seed inputs, mutate them to generate new inputs, and identify the promising inputs using an evolutionary fitness function for further mutation. 

Despite their success, evolutionary fuzzers tend to get stuck in long sequences of unproductive mutations. In recent years, machine learning (ML) based mutation strategies have reported promising results. However, the existing ML-based fuzzers are limited by the lack of quality and diversity of the training data. As the input space of the target programs is high dimensional and sparse, it is prohibitively expensive to collect many diverse samples demonstrating successful and unsuccessful mutations to train the model.

In this paper, we address these issues by using a Multi-Task Neural Network that can learn a compact embedding of the input space based on diverse training samples for multiple related tasks (i.e., predicting different types of coverage). The compact embedding can be used to guide the mutation process effectively by focusing most of the mutations on the parts of the embedding where the gradient is high. Our results show that MTFuzz uncovers 11 previously unseen bugs and achieves an average of 2x more edge coverage compared with 5 state-of-the-art fuzzer on 10 real-world programs.

### Prerequisite
Python 3.7
Tensorflow 1.15.0
Keras 2.2.4

### Usage 
We use readelf as an example
1. Compile programs with 3 different coverage. 
```bash
    CC=ec_pass/afl-clang-fast ./configure && make  # build ec coverage program
    CC=ctx_pass/afl-clang-fast ./configure && make # build ctx coverage program
    CC=approach_pass/afl-clang-fast ./configure && make # build approach level coverage program
```
2. Compile programs to intercept operands of every CMP instrutions.
```bash
   CC=br_pass/afl-clang-fast ./configure && make # instrment every CMP instutions of program 
   CC=br_fast_pass/afl-clang-fast ./configure && make # faster version using fork server 
```
3. Run multi-task nn module.
```bash
   python ./nn.py ./readelf -a 
```
4. Run fuzzing module.
```bash
   python ./mtfuzz_wrapper.py -i mtfuzz_in -o seeds -l 7406 ./readelf -a @@

```

### Tested programs
We provide 10 real world programs and training datasets for reproduce our results.

### Contant
Feel free to send me email about MTFuzz. dongdong at cs.columbia.edu


