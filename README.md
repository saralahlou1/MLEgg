# MLEgg
## Applying Equality Saturation to MLIR
Compiler optimization plays a crucial role in improving code efficiency. However, generating optimized code is a complex task beyond simple code translation. Specialized intermediate representations, such as those offered by MLIR, can help preserve and represent different programming paradigms at a high level. Despite the potential benefits, effectively using these intermediate representations for code optimization often requires a manual and destructive process. 
<br />
The reason for this is that optimizations are applied sequentially. Each optimization destructively modifies the program to produce a transformed program, which is then passed to the next optimization. Depending on the order in which the optimizations are applied, we may not achieve the best outcome. To address this issue, equality saturation, a technique that simultaneously represents multiple program versions, efficiently optimizes the generated code by programmatically generating and representing equivalent statements. These statements are inferred following user-defined rules specific to the language. The user can also define a cost model specific to the language to choose the best expression. 
<br />
This project is an attempt at implementing an MLIR pass that applies equality saturation to a simple MLIR program. It can parse and interpret instructions, leveraging an efficient equality saturation runner to generate alternate, more optimized representations of the program structure, enabling the automatic application of optimizations without ordering constraints. This project focuses on the Linalg dialect and implements a cost model to help us choose the best version of the program.

## MLIR version
```sh
LLVM (http://llvm.org/):
  LLVM version 17.0.3
  DEBUG build with assertions.
```
