; ModuleID = './test-2.c'
source_filename = "./test-2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@main.comment1 = private unnamed_addr constant [27 x i8] c"Initilize a comment string\00", align 16
@main.comment2 = private unnamed_addr constant [26 x i8] c"Set an intermediate value\00", align 16
@main.comment3 = private unnamed_addr constant [50 x i8] c"If the argument is larger than 1000 call func_a()\00", align 16
@main.comment4 = private unnamed_addr constant [48 x i8] c"If the argument is less than 1000 call func_b()\00", align 16
@main.comment5 = private unnamed_addr constant [14 x i8] c"Call func_c()\00", align 1
@main.comment6 = private unnamed_addr constant [14 x i8] c"Call func_d()\00", align 1
@main.comment7 = private unnamed_addr constant [8 x i8] c"The end\00", align 1
@__afl_area_ptr = external global i8*
@__afl_prev_loc = external thread_local global i32

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %0 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %1 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %2 = xor i32 %0, 48786
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = load i8, i8* %3, !nosanitize !2
  %5 = add i8 %4, 1
  store i8 %5, i8* %3, !nosanitize !2
  store i32 24393, i32* @__afl_prev_loc, !nosanitize !2
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %arg_1 = alloca i32, align 4
  %comment1 = alloca [27 x i8], align 16
  %comment2 = alloca [26 x i8], align 16
  %intermediate_val_1 = alloca i32, align 4
  %comment3 = alloca [50 x i8], align 16
  %comment4 = alloca [48 x i8], align 16
  %comment5 = alloca [14 x i8], align 1
  %intermediate_val_2 = alloca i32, align 4
  %comment6 = alloca [14 x i8], align 1
  %intermediate_val_3 = alloca i32, align 4
  %comment7 = alloca [8 x i8], align 1
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  %6 = load i8**, i8*** %argv.addr, align 8
  %arrayidx = getelementptr inbounds i8*, i8** %6, i64 1
  %7 = load i8*, i8** %arrayidx, align 8
  %call = call i32 @atoi(i8* %7) #3
  store i32 %call, i32* %arg_1, align 4
  %8 = bitcast [27 x i8]* %comment1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %8, i8* align 16 getelementptr inbounds ([27 x i8], [27 x i8]* @main.comment1, i32 0, i32 0), i64 27, i1 false)
  %9 = bitcast [26 x i8]* %comment2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %9, i8* align 16 getelementptr inbounds ([26 x i8], [26 x i8]* @main.comment2, i32 0, i32 0), i64 26, i1 false)
  %10 = load i32, i32* %arg_1, align 4
  %cmp = icmp sge i32 %10, 1000
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %11 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %12 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %13 = xor i32 %11, 58048
  %14 = getelementptr i8, i8* %12, i32 %13
  %15 = load i8, i8* %14, !nosanitize !2
  %16 = add i8 %15, 1
  store i8 %16, i8* %14, !nosanitize !2
  store i32 29024, i32* @__afl_prev_loc, !nosanitize !2
  %17 = bitcast [50 x i8]* %comment3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %17, i8* align 16 getelementptr inbounds ([50 x i8], [50 x i8]* @main.comment3, i32 0, i32 0), i64 50, i1 false)
  %18 = load i32, i32* %arg_1, align 4
  %call1 = call i32 @func_a()
  store i32 %call1, i32* %intermediate_val_1, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %19 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %20 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %21 = xor i32 %19, 26103
  %22 = getelementptr i8, i8* %20, i32 %21
  %23 = load i8, i8* %22, !nosanitize !2
  %24 = add i8 %23, 1
  store i8 %24, i8* %22, !nosanitize !2
  store i32 13051, i32* @__afl_prev_loc, !nosanitize !2
  %25 = bitcast [48 x i8]* %comment4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %25, i8* align 16 getelementptr inbounds ([48 x i8], [48 x i8]* @main.comment4, i32 0, i32 0), i64 48, i1 false)
  %26 = load i32, i32* %arg_1, align 4
  %call2 = call i32 @func_b()
  store i32 %call2, i32* %intermediate_val_1, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %27 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %28 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %29 = xor i32 %27, 18073
  %30 = getelementptr i8, i8* %28, i32 %29
  %31 = load i8, i8* %30, !nosanitize !2
  %32 = add i8 %31, 1
  store i8 %32, i8* %30, !nosanitize !2
  store i32 9036, i32* @__afl_prev_loc, !nosanitize !2
  %33 = bitcast [14 x i8]* %comment5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %33, i8* align 1 getelementptr inbounds ([14 x i8], [14 x i8]* @main.comment5, i32 0, i32 0), i64 14, i1 false)
  %34 = load i32, i32* %intermediate_val_1, align 4
  %call3 = call i32 @func_c(i32 %34)
  store i32 %call3, i32* %intermediate_val_2, align 4
  %35 = bitcast [14 x i8]* %comment6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %35, i8* align 1 getelementptr inbounds ([14 x i8], [14 x i8]* @main.comment6, i32 0, i32 0), i64 14, i1 false)
  %36 = load i32, i32* %intermediate_val_2, align 4
  %call4 = call i32 @func_d(i32 %36)
  store i32 %call4, i32* %intermediate_val_3, align 4
  %37 = bitcast [8 x i8]* %comment7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %37, i8* align 1 getelementptr inbounds ([8 x i8], [8 x i8]* @main.comment7, i32 0, i32 0), i64 8, i1 false)
  ret i32 0
}

; Function Attrs: nounwind readonly
declare dso_local i32 @atoi(i8*) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @func_c(i32 %val) #0 {
entry:
  %0 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %1 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %2 = xor i32 %0, 15224
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = load i8, i8* %3, !nosanitize !2
  %5 = add i8 %4, 1
  store i8 %5, i8* %3, !nosanitize !2
  store i32 7612, i32* @__afl_prev_loc, !nosanitize !2
  %val.addr = alloca i32, align 4
  store i32 %val, i32* %val.addr, align 4
  %6 = load i32, i32* %val.addr, align 4
  %mul = mul nsw i32 %6, 5
  ret i32 %mul
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @func_d(i32 %val) #0 {
entry:
  %0 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %1 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %2 = xor i32 %0, 58813
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = load i8, i8* %3, !nosanitize !2
  %5 = add i8 %4, 1
  store i8 %5, i8* %3, !nosanitize !2
  store i32 29406, i32* @__afl_prev_loc, !nosanitize !2
  %val.addr = alloca i32, align 4
  store i32 %val, i32* %val.addr, align 4
  %6 = load i32, i32* %val.addr, align 4
  %div = sdiv i32 %6, 10
  ret i32 %div
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @func_a() #0 {
entry:
  %0 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %1 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %2 = xor i32 %0, 28446
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = load i8, i8* %3, !nosanitize !2
  %5 = add i8 %4, 1
  store i8 %5, i8* %3, !nosanitize !2
  store i32 14223, i32* @__afl_prev_loc, !nosanitize !2
  ret i32 10000
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @func_b() #0 {
entry:
  %0 = load i32, i32* @__afl_prev_loc, !nosanitize !2
  %1 = load i8*, i8** @__afl_area_ptr, !nosanitize !2
  %2 = xor i32 %0, 15464
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = load i8, i8* %3, !nosanitize !2
  %5 = add i8 %4, 1
  store i8 %5, i8* %3, !nosanitize !2
  store i32 7732, i32* @__afl_prev_loc, !nosanitize !2
  ret i32 -10000
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind readonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 "}
!2 = !{}
