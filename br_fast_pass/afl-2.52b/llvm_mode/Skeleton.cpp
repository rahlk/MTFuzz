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

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {

  class AFLCoverage : public ModulePass {

    public:

      static char ID;
      AFLCoverage() : ModulePass(ID) { }

      bool runOnModule(Module &M) override;

      // StringRef getPassName() const override {
      //  return "American Fuzzy Lop Instrumentation";
      // }

  };

}


char AFLCoverage::ID = 0;


bool AFLCoverage::runOnModule(Module &M) {

  LLVMContext &C = M.getContext();

  IntegerType *Int1Ty  = IntegerType::getInt1Ty(C);
  IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  IntegerType *Int16Ty  = IntegerType::getInt16Ty(C);
  IntegerType *Int64Ty  = IntegerType::getInt64Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);
  Type *retType = Type::getVoidTy(C);
  PointerType *Int8Ptr = PointerType::get(Int8Ty,0);

  std::vector<Type*> paramTypes_1 = {Type::getInt32Ty(C), Type::getInt8Ty(C), Type::getInt8Ty(C), Type::getInt32Ty(C)};
  FunctionType *logFuncType_1 = FunctionType::get(retType, paramTypes_1, false);
  Constant *check_br8 = (&M)->getOrInsertFunction("check_br8", logFuncType_1);
 
  std::vector<Type*> paramTypes_2 = {Type::getInt32Ty(C), Type::getInt16Ty(C), Type::getInt16Ty(C), Type::getInt32Ty(C)};
  FunctionType *logFuncType_2 = FunctionType::get(retType, paramTypes_2, false);
  Constant *check_br16 = (&M)->getOrInsertFunction("check_br16", logFuncType_2);
 
  std::vector<Type*> paramTypes_3 = {Type::getInt32Ty(C), Type::getInt32Ty(C), Type::getInt32Ty(C), Type::getInt32Ty(C)};
  FunctionType *logFuncType_3 = FunctionType::get(retType, paramTypes_3, false);
  Constant *check_br32 = (&M)->getOrInsertFunction("check_br32", logFuncType_3);
 
  std::vector<Type*> paramTypes_4 = {Type::getInt32Ty(C), Type::getInt64Ty(C), Type::getInt64Ty(C), Type::getInt32Ty(C)};
  FunctionType *logFuncType_4 = FunctionType::get(retType, paramTypes_4, false);
  Constant *check_br64 = (&M)->getOrInsertFunction("check_br64", logFuncType_4);
  
  std::vector<Type*> paramTypes_5 = {Type::getInt32Ty(C), Type::getInt32Ty(C), Int8Ptr, Int8Ptr,Int32Ty, Type::getInt32Ty(C)};
  FunctionType *logFuncType_5 = FunctionType::get(retType, paramTypes_5, false);
  Constant *check_strcmp = (&M)->getOrInsertFunction("check_strcmp", logFuncType_5);
 
  std::vector<Type*> paramTypes_6 = {Type::getInt32Ty(C), Type::getInt32Ty(C), Int8Ptr, Int8Ptr, Int32Ty, Int32Ty, Type::getInt32Ty(C)};
  FunctionType *logFuncType_6 = FunctionType::get(retType, paramTypes_6, false);
  Constant *check_strncmp = (&M)->getOrInsertFunction("check_strncmp", logFuncType_6);
 
  
  /*

  
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
  int cnt = 0; 
  for (auto &F : M){
    for (auto &BB : F){
      for (auto &I :BB){ 
        if(CmpInst* cmp_inst = dyn_cast<CmpInst>(&I)){
          if(cmp_inst->getNumOperands() == 2){
            Value *op1 = cmp_inst->getOperand(0);
            Value *op2 = cmp_inst->getOperand(1);
            Value* constant_loc =  ConstantInt::get(Int32Ty, 0);
            int constantLoc = 0;
            if(isa<Constant>(op1) && !isa<Constant>(op2)){
                constant_loc = ConstantInt::get(Int32Ty, 1);
                constantLoc = 1;
            }
            else if (!isa<Constant>(op1) && isa<Constant>(op2)){
                constant_loc = ConstantInt::get(Int32Ty, 2);
                constantLoc = 2;
            }
            if(auto* type = dyn_cast<IntegerType>(op1->getType())){
              cnt = cnt + 1;
              switch(type->getBitWidth()){
                  case 8:
                      {
                      IRBuilder<> IRB(cmp_inst--); 
                      Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                      Value* args[] = {br_id, op1, op2, constant_loc};  
                      IRB.CreateCall(check_br8,args);
                      errs()<<"$$$### "<< cnt << " " << constantLoc << "\n";
                      }
                      break;
                  case 16:
                      {
                      IRBuilder<> IRB(cmp_inst--); 
                      Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                      Value* args[] = {br_id, op1, op2, constant_loc};  
                      IRB.CreateCall(check_br16,args);
                      errs()<<"$$$### "<< cnt << " " << constantLoc << "\n";
                      }
                      break;
                  case 32:
                      {
                      IRBuilder<> IRB(cmp_inst--); 
                      Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                      Value* args[] = {br_id, op1, op2, constant_loc};  
                      IRB.CreateCall(check_br32,args);
                      errs()<<"$$$### "<< cnt << " " << constantLoc << "\n";
                      }
                      break;
                  case 64:
                      {
                      IRBuilder<> IRB(cmp_inst--); 
                      Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                      Value* args[] = {br_id, op1, op2, constant_loc};  
                      IRB.CreateCall(check_br64,args);
                      errs()<<"$$$### "<< cnt << " " << constantLoc << "\n";
                      }
                      break;
                  default:
                      break;
              }
            }
          }
        } 
        else if(auto* sw_inst = dyn_cast<SwitchInst>(&I)){
          Value* op1 = sw_inst->getCondition();
          if(auto* type = dyn_cast<IntegerType>(op1->getType())){
            SmallVector<ConstantInt*,128> *case_val_list=new SmallVector<ConstantInt*,128>();
            switch(type->getBitWidth()){
              case 8:
                  {
                  for(auto i = sw_inst->case_begin(), e = sw_inst->case_end(); i != e;++i){
                    ConstantInt* op2 = dyn_cast<ConstantInt>(i->getCaseValue()); 
                    case_val_list->push_back(op2); 
                  }
                  int num_cases = sw_inst->getNumCases();
                  IRBuilder<> IRB(sw_inst--); 
                  for (int i = 0; i< num_cases;i++){
                    cnt = cnt + 1;
                    Value* op2 = case_val_list->pop_back_val();  
                    Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                    Value* constant_loc =  ConstantInt::get(Int32Ty, 2); 
                    Value* args[] = {br_id, op1, op2, constant_loc};  
                    IRB.CreateCall(check_br8,args); 
                    errs()<<"$$$### "<< cnt << " 11 " << 2 << "\n";
                  }
                  }
                  break;
              case 16:
                  {
                  for(auto i = sw_inst->case_begin(), e = sw_inst->case_end(); i != e;++i){
                    ConstantInt* op2 = dyn_cast<ConstantInt>(i->getCaseValue()); 
                    case_val_list->push_back(op2); 
                  }
                  int num_cases = sw_inst->getNumCases();
                  IRBuilder<> IRB(sw_inst--); 
                  for (int i = 0; i< num_cases;i++){
                    cnt = cnt + 1;
                    Value* op2 = case_val_list->pop_back_val();  
                    Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                    Value* constant_loc =  ConstantInt::get(Int32Ty, 2); 
                    Value* args[] = {br_id, op1, op2, constant_loc};  
                    IRB.CreateCall(check_br16,args); 
                    errs()<<"$$$### "<< cnt << " 11 " << 2 << "\n";
                  }
                  }
                  break;
              case 32:
                  {
                  for(auto i = sw_inst->case_begin(), e = sw_inst->case_end(); i != e;++i){
                    ConstantInt* op2 = dyn_cast<ConstantInt>(i->getCaseValue()); 
                    case_val_list->push_back(op2); 
                  }
                  int num_cases = sw_inst->getNumCases();
                  IRBuilder<> IRB(sw_inst--); 
                  for (int i = 0; i< num_cases;i++){
                    cnt = cnt + 1;
                    Value* op2 = case_val_list->pop_back_val();  
                    Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                    Value* constant_loc =  ConstantInt::get(Int32Ty, 2); 
                    Value* args[] = {br_id, op1, op2, constant_loc};  
                    IRB.CreateCall(check_br32,args); 
                    errs()<<"$$$### "<< cnt << " 11 " << 2 << "\n";
                  }
                  }
                  break;
              case 64:
                  {
                  for(auto i = sw_inst->case_begin(), e = sw_inst->case_end(); i != e;++i){
                    ConstantInt* op2 = dyn_cast<ConstantInt>(i->getCaseValue()); 
                    case_val_list->push_back(op2); 
                  }
                  int num_cases = sw_inst->getNumCases();
                  IRBuilder<> IRB(sw_inst--); 
                  for (int i = 0; i< num_cases;i++){
                    cnt = cnt + 1;
                    Value* op2 = case_val_list->pop_back_val();  
                    Value* br_id =  ConstantInt::get(Int32Ty, cnt);
                    Value* constant_loc =  ConstantInt::get(Int32Ty, 2); 
                    Value* args[] = {br_id, op1, op2, constant_loc}; 
                    IRB.CreateCall(check_br64,args); 
                    errs()<<"$$$### "<< cnt << " 11 " << 2 << "\n";
                  }
                  }
                  break;
              default:
                  break;
            }
          }
        }
        else if(auto* call_inst = dyn_cast<CallInst>(&I)){
          if(Function *fun = call_inst->getCalledFunction()){
            if(fun->getName().equals("strcmp")){
              cnt = cnt + 1;
              Value* br_id =  ConstantInt::get(Int32Ty, cnt);
              Value* op1 = call_inst->getArgOperand(0);
              Value* op2 = call_inst->getArgOperand(1);
              Value* type =  ConstantInt::get(Int32Ty, 10);
              Value* constant_loc =  ConstantInt::get(Int32Ty, 0);
              Value* ret =  dyn_cast<Value>(call_inst);
              if(auto hope = dyn_cast<ConstantExpr>(op1)){
                constant_loc =  ConstantInt::get(Int32Ty, 1);
              }
              if(auto hope = dyn_cast<ConstantExpr>(op2)){
                constant_loc =  ConstantInt::get(Int32Ty, 2);
              }

              IRBuilder<> IRB(call_inst->getNextNode());
              Value* args[] = {br_id, type, op1, op2,ret, constant_loc};
              IRB.CreateCall(check_strcmp, args);

              /*
              for(auto arg = call_inst->arg_begin(); arg != call_inst->arg_end(); ++arg) {
                if(auto hope = dyn_cast<ConstantExpr>(arg)){
                  auto glo_var = dyn_cast<GlobalVariable>(hope->getOperand(0));
                  auto const_data = dyn_cast<ConstantDataArray>(glo_var->getInitializer());
                  StringRef strr = const_data->getAsCString();
                }
              }*/

            }
            if(fun->getName().equals("strncmp")){
              cnt = cnt + 1;
              Value* br_id =  ConstantInt::get(Int32Ty, cnt);
              Value* op1 = call_inst->getArgOperand(0);
              Value* op2 = call_inst->getArgOperand(1);
              Value* len = call_inst->getArgOperand(2);
              Value* type =  ConstantInt::get(Int32Ty, 12);
              Value* constant_loc =  ConstantInt::get(Int32Ty, 0);
              Value* ret =  dyn_cast<Value>(call_inst);
              if(auto hope = dyn_cast<ConstantExpr>(op1)){
                constant_loc =  ConstantInt::get(Int32Ty, 1);
              }
              if(auto hope = dyn_cast<ConstantExpr>(op2)){
                constant_loc =  ConstantInt::get(Int32Ty, 2);
              }
              IRBuilder<> IRB(call_inst->getNextNode());
              Value* args[] = {br_id, type, op1, op2, len, ret, constant_loc};
              IRB.CreateCall(check_strncmp, args);

              /*
              for(auto arg = call_inst->arg_begin(); arg != call_inst->arg_end(); ++arg) {
                if(auto hope = dyn_cast<ConstantExpr>(arg)){
                  auto glo_var = dyn_cast<GlobalVariable>(hope->getOperand(0));
                  auto const_data = dyn_cast<ConstantDataArray>(glo_var->getInitializer());
                  StringRef strr = const_data->getAsCString();
                }
              }*/
            }
          }
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

}


static RegisterStandardPasses RegisterAFLPass(
    PassManagerBuilder::EP_OptimizerLast, registerAFLPass);

static RegisterStandardPasses RegisterAFLPass0(
    PassManagerBuilder::EP_EnabledOnOptLevel0, registerAFLPass);
