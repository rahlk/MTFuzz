/*  LLVM instrumentation for context sensitive branch coverage.
 *
 *  Code extened from AFL instrumentation. See: github.com/mirrorer/afl
 */
#define AFL_LLVM_PASS

#include "../config.h"
#include "../debug.h"
#include "../types.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

using namespace llvm;
using namespace std;

/*
 * VANILLA AFL INSTRUMENTATION
 */

namespace {
class AFLCoverage : public ModulePass {
public:
  static char ID;
  AFLCoverage() : ModulePass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};
} // namespace

char AFLCoverage::ID = 0;
bool AFLCoverage::runOnModule(Module &M) {
  srandom(12);
  LLVMContext &C = M.getContext();

  IntegerType *Int8Ty = IntegerType::getInt8Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);

  /* Show a banner */
  char be_quiet = 0;

  if (isatty(2) && !getenv("AFL_QUIET")) {

    SAYF(cCYA "afl-llvm-pass " cBRI VERSION cRST
              " by <lszekeres@google.com>\n");
  } else
    be_quiet = 1;

  /* Decide instrumentation ratio */
  char *inst_ratio_str = getenv("AFL_INST_RATIO");
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
      M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_prev_loc", 0,
      GlobalVariable::GeneralDynamicTLSModel, 0, false);
  
  GlobalVariable *AFLCurLoc = new GlobalVariable(
      M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_cur_loc", 0,
      GlobalVariable::GeneralDynamicTLSModel, 0, false);

  /* Instrument all the things! */
  int inst_blocks = 0;

  for (auto &F : M)
    for (auto &BB : F) {
      BasicBlock::iterator IP = BB.getFirstInsertionPt();
      IRBuilder<> IRB(&(*IP));

      if (AFL_R(100) >= inst_ratio)
        continue;

      /* Make up cur_loc */
      unsigned int cur_loc = AFL_R(MAP_SIZE);

      ConstantInt *CurLoc = ConstantInt::get(Int32Ty, cur_loc);

      /* Load prev_loc */
      LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
      PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());

      /* Load SHM pointer */
      LoadInst *MapPtr = IRB.CreateLoad(AFLMapPtr);
      MapPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *MapPtrIdx =
          IRB.CreateGEP(MapPtr, IRB.CreateXor(PrevLocCasted, CurLoc));

      /* Update bitmap */
      LoadInst *Counter = IRB.CreateLoad(MapPtrIdx);
      Counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *Incr = IRB.CreateAdd(Counter, ConstantInt::get(Int8Ty, 1));
      IRB.CreateStore(Incr, MapPtrIdx)
          ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

      
      /* Set prev_loc to cur_loc >> 1 */
      StoreInst *Store =
          IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), AFLPrevLoc);
      Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      
      Store =
          IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc), AFLCurLoc);
      Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

      inst_blocks++;
    }

  /* Say something nice. */

  if (!be_quiet) {

    if (!inst_blocks)
      WARNF("No instrumentation targets found.");
    else
      OKF("Instrumented %u locations (%s mode, ratio %u%%).", inst_blocks,
          getenv("AFL_HARDEN")
              ? "hardened"
              : ((getenv("AFL_USE_ASAN") || getenv("AFL_USE_MSAN"))
                     ? "ASAN/MSAN"
                     : "non-hardened"),
          inst_ratio);
  }

  return true;
}

/*
 * CONTEXT SENSITIVE BRANCH COVERAGE INSTRUMENTATION
 */

namespace {

class CtxSenCov : public ModulePass {

public:
  static char ID;
  CtxSenCov() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AFLCoverage>();
  }
  bool runOnModule(Module &M) override;
};

} // namespace

char CtxSenCov::ID = 0;

bool CtxSenCov::runOnModule(Module &M) {
  // Set random number seed
  srandom(12);

  LLVMContext &C = M.getContext();

  // Create a map to hold the function call site and the random ID
  map<Value *, unsigned int> func_id_map;

  /* Define types
     ------------ */
  IntegerType *Int8Ty = IntegerType::getInt8Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);

  /* Initialize global variables
     --------------------------- */
  // Pointer to AFL Share Memory Bitmap
  GlobalVariable *AFLMapPtr = M.getGlobalVariable("__afl_area_ptr");

  
  // Pointer to AFL CurLoc
  GlobalVariable *AFLCurLoc = M.getGlobalVariable("__afl_cur_loc");

  // Pointer to AFL Call Context
  GlobalVariable *AFLCallCtx = new GlobalVariable(
      M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_call_ctx", 0,
      GlobalVariable::GeneralDynamicTLSModel, 0, false);


  // -----------------------
  /* Begin instrumentation
     ----------------------- */
  int inst_count = 0;
  for (auto &F : M) {

    /* Do something only if it is not an external/inbuilt function */
    //if (F.hasExactDefinition()) {
      for (auto &B : F) {
        for (auto &I : B) {
          /* Loop through every instruction. */
          IRBuilder<> builder(&I);

          /* To avoid "PHI nodes not grouped at top of basic block." */
          if (isa<PHINode>(I)) {
            continue;
          }

          /* Check to see if we have reached a 'call' instruction */
          if (auto *CallInstr = dyn_cast<CallInst>(&I)) {
            /* Make sure the call Instruction is not to a builtin function,
             * an intrinsic function, or an indirect call. */
            Function *called_func = CallInstr->getCalledFunction();

            if (!called_func || called_func->isIntrinsic() ||
                !called_func->hasExactDefinition()) {
              continue;
            } else {
              /* Update call_context above and below a the call site. */
              /* Get Call site (which is the address of the current
               * instruction).*/
                
              /* Get the current insertion point */
              BasicBlock::iterator insert_pt_above = builder.GetInsertPoint();
              BasicBlock::iterator insert_pt_below = builder.GetInsertPoint();

              /* ==============================================================
               */
              // ----- Above call site -----
              // -- Set Instruction pointer --
              inst_count++;
              builder.SetInsertPoint(&B, insert_pt_above);

              /* Load SHM pointer */
              LoadInst *MapPtr = builder.CreateLoad(AFLMapPtr);

              // -- Load cur_loc --
              LoadInst *CurEdgeLoc = builder.CreateLoad(AFLCurLoc);
              Value *CurEdgeCasted = builder.CreateZExt(CurEdgeLoc, builder.getInt32Ty());

              // -- Load call context --
              LoadInst *CallCtx = builder.CreateLoad(AFLCallCtx);
              Value *CallCtxCasted =
                  builder.CreateZExt(CallCtx, builder.getInt32Ty());

              int call_id = AFL_R(MAP_SIZE);
              // -- Update call context with an XOR --
              ConstantInt *CurFuncId =
                  ConstantInt::get(Int32Ty, call_id);
              Value *callctx_updated =
                  builder.CreateXor(CurFuncId, CallCtxCasted);

              // Update the global call context variable
              builder.CreateStore(callctx_updated, AFLCallCtx);

              // -- Update vanilla Map Pointer index --
              Value *MapPtrIdx_branch_id_w_callctx =
                  builder.CreateXor(CurEdgeCasted, callctx_updated);

              Value *MapPtrIdx =
                  builder.CreateGEP(MapPtr, MapPtrIdx_branch_id_w_callctx);

              // -- Increment counter and update AFL MAP --
              LoadInst *Counter = builder.CreateLoad(MapPtrIdx);
              Value *Incr =
                  builder.CreateAdd(Counter, ConstantInt::get(Int8Ty, 1));
              builder.CreateStore(Incr, MapPtrIdx);

              /* ============================================================== */
              // ----- Below call site -----
              // -- Set Instruction pointer --
              builder.SetInsertPoint(&B, ++insert_pt_below);

              callctx_updated = builder.CreateXor(CurFuncId, callctx_updated);

              // Update the global call context variable
              builder.CreateStore(callctx_updated, AFLCallCtx);

            }
          }
        }
      }
    }
  if (VERBOSE) {
    if (!inst_count)
      WARNF("No function targets found.");
    else
      OKF("Instrumented %u functions.", inst_count);
  }
  return true;
}

static void registerAFLPass(const PassManagerBuilder &,
                            legacy::PassManagerBase &PM) {
  PM.add(new AFLCoverage());
  PM.add(new CtxSenCov());
}

static RegisterStandardPasses
    RegisterAFLPass(PassManagerBuilder::EP_OptimizerLast, registerAFLPass);

static RegisterStandardPasses
    RegisterAFLPass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                     registerAFLPass);
