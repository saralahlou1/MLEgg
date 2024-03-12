module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @square(%arg0: i32 {llvm.noundef}) -> i32 attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["noinline", "nounwind", "optnone", "ssp", ["uwtable", "1"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["probe-stack", "__chkstk_darwin"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+crc", "+crypto", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+sm4", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.mul %2, %3 overflow<nsw>  : i32
    llvm.return %4 : i32
  }
  llvm.func @main() -> i32 attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["noinline", "nounwind", "optnone", "ssp", ["uwtable", "1"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["probe-stack", "__chkstk_darwin"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+crc", "+crypto", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+sm4", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %2, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %8 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.call @square(%8) : (i32) -> i32
    llvm.store %9, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    %10 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.call @printf(%4, %10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    llvm.return %1 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = [["no-trapping-math", "true"], ["probe-stack", "__chkstk_darwin"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+crc", "+crypto", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+sm4", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}
}
