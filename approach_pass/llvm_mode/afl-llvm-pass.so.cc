/*
   american fuzzy lop - LLVM-mode instrumentation pass
   ---------------------------------------------------

   Written by Laszlo Szekeres <lszekeres@google.com> and
              Michal Zalewski <lcamtuf@google.com>

   LLVM integration design comes from Laszlo Szekeres. C bits copied-and-pasted
   from afl-as.c are Michal's fault.

   Copyright 2015, 2016 Google Inc. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     http://www.apache.org/licenses/LICENSE-2.0

   This library is plugged into LLVM when invoking clang through afl-clang-fast.
   It tells the compiler to add code roughly equivalent to the bits discussed
   in ../afl-as.h.

 */

#define AFL_LLVM_PASS

#include "../config.h"
#include "../debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "llvm/Analysis/LoopInfo.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Module.h"



using namespace llvm;

namespace {

  class AFLCoverage : public ModulePass {

    public:

      static char ID;
      AFLCoverage() : ModulePass(ID) { }

      void getAnalysisUsage(AnalysisUsage &AU) const override{
        AU.addRequired<LoopInfoWrapperPass>();
      }
      
      bool runOnModule(Module &M) override;
    
      // StringRef getPassName() const override {
      //  return "American Fuzzy Lop Instrumentation";
      // }

  };

}


char AFLCoverage::ID = 0;


bool AFLCoverage::runOnModule(Module &M) {

  srandom(12);
  LLVMContext &C = M.getContext();

  IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  IntegerType *Int64Ty  = IntegerType::getInt64Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);
  
  Type *retType = Type::getVoidTy(C);

  std::vector<Type*> paramTypes_5 = {Type::getInt64Ty(C)};
  FunctionType *logFuncType_5 = FunctionType::get(retType, paramTypes_5, false);
  Constant *log_br = (&M)->getOrInsertFunction("log_br", logFuncType_5);

  /* Show a banner */

  char be_quiet = 0;

  if (isatty(2) && !getenv("AFL_QUIET")) {

    SAYF(cCYA "afl-llvm-pass " cBRI VERSION cRST " by <lszekeres@google.com>\n");

  } else be_quiet = 1;

  /* Decide instrumentation ratio */

  char* inst_ratio_str = getenv("AFL_INST_RATIO");
  unsigned int inst_ratio = 100;

  if (inst_ratio_str) {

    if (sscanf(inst_ratio_str, "%u", &inst_ratio) != 1 || !inst_ratio ||
        inst_ratio > 100)
      FATAL("Bad value of AFL_INST_RATIO (must be between 1 and 100)");

  }

  /* Get globals for the SHM region and the previous location. Note that
     __afl_prev_loc is thread-local. */

  GlobalVariable *AFLMapPtr =
      new GlobalVariable(M, PointerType::get(Int8Ty, 0), false,
                         GlobalValue::ExternalLinkage, 0, "__afl_area_ptr");

  
  GlobalVariable *AFLPrevLoc = new GlobalVariable(
      M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_prev_loc",
      0, GlobalVariable::GeneralDynamicTLSModel, 0, false);

  /* Instrument all the things! */

  int inst_blocks = 0;

  for(auto &F : M){    
    for (auto &BB : F) {
      BasicBlock::iterator IP = BB.getFirstInsertionPt();
      IRBuilder<> IRB(&(*IP));

      if (AFL_R(100) >= inst_ratio) continue;

      unsigned int cur_loc = AFL_R(MAP_SIZE);

      ConstantInt *CurLoc = ConstantInt::get(Int32Ty, cur_loc);


      LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
      PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());


      LoadInst *MapPtr = IRB.CreateLoad(AFLMapPtr);
      MapPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *MapPtrIdx =
          IRB.CreateGEP(MapPtr, IRB.CreateXor(PrevLocCasted, CurLoc));


      LoadInst *Counter = IRB.CreateLoad(MapPtrIdx);
      Counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *Incr = IRB.CreateAdd(Counter, ConstantInt::get(Int8Ty, 1));
      IRB.CreateStore(Incr, MapPtrIdx)
          ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      

      StoreInst *Store =
          IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), AFLPrevLoc);
      Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

      inst_blocks++;
     
    } 
  }

  /* Say something nice. */

  if (!be_quiet) {

    if (!inst_blocks) WARNF("No instrumentation targets found.");
    else OKF("Instrumented %u locations (%s mode, ratio %u%%).",
             inst_blocks, getenv("AFL_HARDEN") ? "hardened" :
             ((getenv("AFL_USE_ASAN") || getenv("AFL_USE_MSAN")) ?
              "ASAN/MSAN" : "non-hardened"), inst_ratio);

  }

  return true;

}

namespace {

  class BrLog : public ModulePass {

    public:

      static char ID;
      BrLog() : ModulePass(ID) { }

      void getAnalysisUsage(AnalysisUsage &AU) const override{
        AU.addRequired<AFLCoverage>();
        AU.addRequired<LoopInfoWrapperPass>();
      }
      
      bool runOnModule(Module &M) override;
    
  };

}


char BrLog::ID = 1;


bool BrLog::runOnModule(Module &M) {

  srandom(12);
  LLVMContext &C = M.getContext();

  IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  IntegerType *Int64Ty  = IntegerType::getInt64Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);

  GlobalVariable *AFLMapPtr = M.getGlobalVariable("__afl_area_ptr");

  GlobalVariable *AFLPrevLoc = M.getGlobalVariable("__afl_prev_loc");
  Type *retType = Type::getVoidTy(C);

  std::vector<Type*> paramTypes_5 = {Type::getInt32Ty(C), Int32Ty};
  FunctionType *logFuncType_5 = FunctionType::get(retType, paramTypes_5, false);
  Constant *log_pair_edge = (&M)->getOrInsertFunction("log_pair_edge", logFuncType_5);
  /* Show a banner */

  char be_quiet = 0;

  if (isatty(2) && !getenv("AFL_QUIET")) {

    SAYF(cCYA "afl-llvm-pass " cBRI VERSION cRST " by <lszekeres@google.com>\n");

  } else be_quiet = 1;

  /* Decide instrumentation ratio */

  char* inst_ratio_str = getenv("AFL_INST_RATIO");
  unsigned int inst_ratio = 100;

  if (inst_ratio_str) {

    if (sscanf(inst_ratio_str, "%u", &inst_ratio) != 1 || !inst_ratio ||
        inst_ratio > 100)
      FATAL("Bad value of AFL_INST_RATIO (must be between 1 and 100)");

  }

  /* Get globals for the SHM region and the previous location. Note that
     __afl_prev_loc is thread-local. */


  /* Instrument all the things! */

  int inst_blocks = 0;

  for(auto &F : M){
    for (auto &BB : F) { 
      if(auto term_inst = BB.getTerminator()){
        if(auto br_inst = dyn_cast<BranchInst>(term_inst)){
          if(br_inst->isConditional()){
            if(br_inst->getNumSuccessors() == 2){
              BasicBlock * bb_op1 = br_inst->getSuccessor(0);
              Instruction* first_inst1 = &*(bb_op1->getFirstInsertionPt());
              
              BasicBlock * bb_op2 = br_inst->getSuccessor(1);
              Instruction* first_inst2 = &*(bb_op2->getFirstInsertionPt());
              if(auto xor_inst1 = dyn_cast<BinaryOperator>(first_inst1->getNextNode()->getNextNode())){
                if(auto xor_inst2 = dyn_cast<BinaryOperator>(first_inst2->getNextNode()->getNextNode())){
                  if((strcmp(xor_inst1->getOpcodeName(), "xor")==0) && (strcmp(xor_inst2->getOpcodeName(), "xor")==0)){
                    Value *xor1_op2 = xor_inst1->getOperand(1);
                    Value *xor2_op2 = xor_inst2->getOperand(1);
                    
                    IRBuilder<>IRB(br_inst);
                    
                    LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
                    PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
                    Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());
                    
                    Value* edge_id1 = IRB.CreateXor(PrevLocCasted, xor1_op2);
                    Value* edge_id2 = IRB.CreateXor(PrevLocCasted, xor2_op2);
                    
                    LoadInst *MapPtr = IRB.CreateLoad(AFLMapPtr);

                    Value *MapPtrIdx1 = IRB.CreateGEP(MapPtr, edge_id1);
                    LoadInst *Counter1 = IRB.CreateLoad(MapPtrIdx1);
                    Value *Incr1 = IRB.CreateAdd(Counter1, ConstantInt::get(Int8Ty, 1));
                    IRB.CreateStore(Incr1, MapPtrIdx1);
                    
                    Value *MapPtrIdx2 = IRB.CreateGEP(MapPtr, edge_id2);
                    LoadInst *Counter2 = IRB.CreateLoad(MapPtrIdx2);
                    Value *Incr2 = IRB.CreateAdd(Counter2, ConstantInt::get(Int8Ty, 1));
                    IRB.CreateStore(Incr2, MapPtrIdx2);
                  }
                }
              }    //int bb_is_loop_entry = LI.isLoopHeader(&BB); 
            }      //errs()<<"BB_is_loop_entry: " << bb_is_loop_entry << "\n";
          }       //if bb_is_loop_entry =  
        }
      }
    } 
  }
  /* Say something nice. */

  if (!be_quiet) {

    if (!inst_blocks) WARNF("No instrumentation targets found.");
    else OKF("Instrumented %u locations (%s mode, ratio %u%%).",
             inst_blocks, getenv("AFL_HARDEN") ? "hardened" :
             ((getenv("AFL_USE_ASAN") || getenv("AFL_USE_MSAN")) ?
              "ASAN/MSAN" : "non-hardened"), inst_ratio);

  }

  return true;

}


static void registerAFLPass(const PassManagerBuilder &,
                            legacy::PassManagerBase &PM) {

  
  PM.add(new AFLCoverage());
  PM.add(new BrLog());

}


static RegisterStandardPasses RegisterAFLPass(
    PassManagerBuilder::EP_OptimizerLast, registerAFLPass);

static RegisterStandardPasses RegisterAFLPass0(
    PassManagerBuilder::EP_EnabledOnOptLevel0, registerAFLPass);
