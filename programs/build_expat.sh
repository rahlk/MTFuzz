# build edge coverage binary
CC=$(pwd)/../../ec_pass/afl-clang-fast ./configure --disable-shared; make -j ; cp xmlwf/xmlwf xmlwf_ec; make clean
# build call context binary
CC=$(pwd)/../../ctx_pass/afl-clang-fast ./configure --disable-shared; make -j ; cp xmlwf/xmlwf xmlwf_ctx; make clean
# build approach level binary
CC=$(pwd)/../../approach_pass/afl-clang-fast ./configure --disable-shared; make -j ; cp xmlwf/xmlwf xmlwf_soft; make clean

# instrment every CMP instutions of program, disable parallel building, use ONLY a single core.
CC=$(pwd)/../../br_pass/src/afl-clang-fast ./configure --disable-shared; make ; cp xmlwf/xmlwf xmlwf_br; rm br_cnt; make clean
# faster version using fork server, use ONLY a sinlge core.
CC=$(pwd)/../../br_fast_pass/src/afl-clang-fast ./configure --disable-shared; make; cp xmlwf/xmlwf xmlwf_br_fast; rm br_cnt; make clean
# save operand information for each CMP instruction
mv br_log xmlwf_br_log
