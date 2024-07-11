// syntax for writing generic op to mlir:
// note that running this code alone wouldn't work
// it needs to be inserted in the corresponding place in the pass

mlir::OpBuilder builder(&(block.back()));

mlir::Location loc = builder.getUnknownLoc();

SmallVector<int64_t, 1> shape = {4};
auto resultType = RankedTensorType::get(shape, builder.getF32Type());


Value inTensor = builder.create<tensor::EmptyOp>(loc, resultType.getShape(), builder.getF32Type());
Value outTensor = builder.create<tensor::EmptyOp>(loc, resultType.getShape(), builder.getF32Type());

SmallVector<AffineMap, 2> indexingMaps = {
    AffineMap::getMultiDimIdentityMap(1, builder.getContext()),
    AffineMap::getMultiDimIdentityMap(1, builder.getContext())
};
SmallVector<utils::IteratorType, 1> iteratorTypes = {utils::IteratorType::parallel};

auto genericOp = builder.create<linalg::GenericOp>(
    loc,
    /*resultTypes=*/outTensor.getType(),
    /*inputs=*/ValueRange{inTensor},
    /*outputs=*/ValueRange{outTensor},
    /*indexingMaps=*/indexingMaps,
    /*iteratorTypes=*/iteratorTypes);

// Define the body of the generic operation
auto &region = genericOp.getRegion();
Block *block0 = builder.createBlock(&region);
auto blockArg = block0->addArgument(builder.getF32Type(), loc);
auto blockArg2 = block0->addArgument(builder.getF32Type(), loc);
builder.setInsertionPointToStart(block0);

// Add 1 to each element
auto one = builder.create<arith::ConstantOp>(loc, builder.getF32Type(), builder.getF32FloatAttr(1.0));
auto result = builder.create<arith::AddFOp>(loc, blockArg, one);
builder.create<linalg::YieldOp>(loc, result->getResult(0));
