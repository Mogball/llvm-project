func @test_pattern_debugger(%arg0: i32) -> i32 {
  %c0 = arith.constant 1 : i32
  %0 = arith.addi %c0, %arg0 : i32
  %1 = arith.addi %0, %c0 : i32
  return %1 : i32
}
