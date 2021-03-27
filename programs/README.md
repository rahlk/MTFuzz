We provide 10 real-world programs for users to reproduce edge coverage results reported in our paper. 

We also provide a sample program to show how to run MTFuzz on users' own programs.
- untar the source code
```bash
tar -axvf expat-2.2.9.tar.bz2
```
- build xmlwf
```bash
cd expat-2.2.9
cp ../build_expat.sh 
./build_expat.sh
cd ..
```
- setup MTFuzz
```bash
./setup_mtfuzz_xmlwf.sh
```

