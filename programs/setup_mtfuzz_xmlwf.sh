#!/bin/bash

# create a working directory
mkdir xmlwf
# copy necessary files to working directory
cp xmllint/afl-* xmlwf
cp xmllint/mtfuzz xmlwf
cp xmllint/mtfuzz_wrapper.py xmlwf
cp xmllint/nn.py xmlwf
mkdir xmlwf/seeds xmlwf/vari_seeds xmlwf/crashes
# xmlwf binaries and seed corpus to working directory
cp -r xmlwf_afl_1hr xmlwf/mtfuzz_in
cp expat-2.2.9/xmlwf_* xmlwf
