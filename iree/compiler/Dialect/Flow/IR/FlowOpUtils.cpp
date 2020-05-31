// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/IR/FlowOpUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

Operation *cloneWithNewResultTypes(Operation *op, TypeRange newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(newResultTypes);
  state.addSuccessors(op->getSuccessors());
  state.addAttributes(op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    state.addRegion();
  }
  Operation *newOp = Operation::create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    newOp->getRegion(i).takeBody(op->getRegion(i));
  }
  return newOp;
}

//------------------------------------------------------------------------------
// ClosureOpDce
//------------------------------------------------------------------------------

ClosureOpDce::ClosureOpDce(Operation *closureOp, Block &entryBlock,
                           unsigned variadicOffset)
    : closureOp(closureOp),
      entryBlock(entryBlock),
      variadicOffset(variadicOffset),
      blockArgReplacements(entryBlock.getNumArguments()) {
  assert(closureOp->getNumOperands() ==
         entryBlock.getNumArguments() + variadicOffset);

  // Build data structure for unused operand elision.
  for (auto it : llvm::enumerate(entryBlock.getArguments())) {
    BlockArgument blockArg = it.value();
    Value opArg = closureOp->getOperand(it.index() + variadicOffset);
    if (blockArg.getUses().empty()) {
      // Not used - Drop.
      needsOperandElision = true;
      blockArgReplacements[it.index()] = BlockArgument();
      continue;
    }
    auto existingIt = argToBlockMap.find(opArg);
    if (existingIt == argToBlockMap.end()) {
      // Not found - Record for deduping.
      argToBlockMap.insert(std::make_pair(opArg, blockArg));
    } else {
      // Found - Replace.
      needsOperandElision = true;
      blockArgReplacements[it.index()] = existingIt->second;
    }
  }

  // Build data structure for unused result elision.
  auto *terminator = entryBlock.getTerminator();
  assert(terminator->getNumOperands() == closureOp->getNumResults());
  for (auto it : llvm::enumerate(terminator->getOperands())) {
    auto result = closureOp->getResult(it.index());
    if (result.getUses().empty()) {
      // Not used - Drop.
      needsResultElision = true;
      continue;
    }

    // Is it duplicated prior?
    bool isDup = false;
    for (int i = 0; i < returnIndexMap.size(); ++i) {
      Value newReturn = terminator->getOperand(returnIndexMap[i]);
      if (it.value() == newReturn) {
        // Duplicated.
        isDup = true;
        needsResultElision = true;
        resultIndexMap.push_back(std::make_pair(it.index(), i));
        break;
      }
    }
    if (isDup) continue;

    // Map as-is.
    resultIndexMap.push_back(std::make_pair(it.index(), returnIndexMap.size()));
    returnIndexMap.push_back(it.index());
  }
}

void ClosureOpDce::elideUnusedOperands(OpBuilder &builder) {
  llvm::SmallVector<Value, 8> newOperands(
      closureOp->operand_begin(), closureOp->operand_begin() + variadicOffset);
  unsigned blockArgIndex = 0;
  for (auto it : llvm::enumerate(blockArgReplacements)) {
    llvm::Optional<BlockArgument> replacement = it.value();
    Value currentOpArg = closureOp->getOperand(it.index() + variadicOffset);
    if (!replacement) {
      // No change.
      newOperands.push_back(currentOpArg);
      blockArgIndex++;
      continue;
    } else if (!replacement.getValue()) {
      // Drop.
      entryBlock.eraseArgument(blockArgIndex);
      continue;
    } else {
      // Replace.
      BlockArgument currentBlockArg = entryBlock.getArgument(blockArgIndex);
      currentBlockArg.replaceAllUsesWith(*replacement);
      entryBlock.eraseArgument(blockArgIndex);
    }
  }

  closureOp->setOperands(newOperands);
}

void ClosureOpDce::elideUnusedResults(OpBuilder &builder, bool eraseOriginal) {
  // Determine the result signature transform needed.
  llvm::SmallVector<Type, 4> newResultTypes;
  for (auto index : returnIndexMap) {
    newResultTypes.push_back(closureOp->getResult(index).getType());
  }

  // Re-allocate the op.
  builder.setInsertionPoint(closureOp);
  Operation *newOp =
      builder.insert(cloneWithNewResultTypes(closureOp, newResultTypes));

  // Remap all returns.
  auto *newTerminator = newOp->getRegion(0).front().getTerminator();
  llvm::SmallVector<Value, 4> newReturns(returnIndexMap.size());
  for (unsigned i = 0, e = returnIndexMap.size(); i < e; ++i) {
    int oldIndex = returnIndexMap[i];
    newReturns[i] = newTerminator->getOperand(oldIndex);
  }
  newTerminator->setOperands(newReturns);

  // Replace original uses.
  for (auto indexMap : resultIndexMap) {
    unsigned oldIndex = indexMap.first;
    unsigned newIndex = indexMap.second;
    Value oldResult = closureOp->getResult(oldIndex);
    Value newResult = newOp->getResult(newIndex);
    oldResult.replaceAllUsesWith(newResult);
  }
  if (eraseOriginal) closureOp->erase();
  closureOp = newOp;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
