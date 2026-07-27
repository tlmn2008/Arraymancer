# Arraymancer → Iluvatar CoreX (ivcore11) 迁移记录

## 来源
- 仓库：mratsim/Arraymancer（Nim 的 n 维张量/ndarray 库）
- 分支/commit：master @ `d3b67712a3334ab4ded2809cc793e9cd7fed5fae`（2026-05-26）
- clone 到 `/home/repos/Arraymancer`（默认分支 HEAD）

## CUDA 使用性质
Arraymancer 有一个可选的 CUDA 后端（`-d:cuda`）：
- 通过 Nim 的 cpp 后端把 `.nim` 生成 `.cpp`，上游用 **nvcc**（把 `.cpp` 当 `--x cu`）编译；
- 线性代数走 **cuBLAS**（`cublasSgemm/Dgemm`、`Sgemv/Dgemv`、`Sdot/Ddot`、`geam`、`gemmStridedBatched` 等，L1/L2/L3）；
- element-wise 运算是 repo 内嵌的 CUDA C++ 模板 kernel（`{.emit.}` 生成 `__device__ __forceinline__` 函数对象 + `<<<>>>` 启动，含 `__ldg`）；
- CUDA API 绑定来自 nimble 依赖 **nimcuda 0.2.2**（`nimcuda/cuda12_5/*`，运行期 `dlopen` 加载 `libcudart/libcublas`）；
- 另有可选 cuDNN 卷积后端（`-d:cudnn`）。

## 环境
- SDK：`/usr/local/corex`（IX-ML 4.4.0 / Driver 4.5.0 / CUDA 10.2 / clang 22.1.0）；
- GPU：2× Iluvatar BI-V150（`ixsmi` 可见，`CUDA_VISIBLE_DEVICES=0,1`，满足 ≤2 卡约束）；
- 工具链：CoreX clang/clang++；额外安装 **Nim 2.2.10**（choosenim，装到 `/root/.nimble`），`nimble install -d` 拉齐依赖；
- CPU 侧 BLAS/LAPACK 用系统 `libblas.so.3`（含 cblas 符号）/`liblapack.so.3`；
- 关键限制遵守：不碰 `/usr/local/corex`、不装/不伪造 nvcc、GPU/Rank ≤2、workaround 全部 repo-local。

## 适配内容
1. **nvcc → CoreX clang++（`nim.cfg` 的 CUDA 段整体重写）**
   - `cc:clang`、`clang.cpp.exe/linkerexe = /usr/local/corex/bin/clang++`；
   - 编译选项 `options.always` 由 `-gencode arch=compute_61 ... --x cu -Xcompiler -fpermissive` 改为 `-x ivcore --cuda-gpu-arch=ivcore11`（探针验证：`-x cuda` 会告警要求换 `-x ivcore`，`-x ivcore` 干净）；
   - `cincludes/clibdir` 指向 `/usr/local/corex/{include,lib64}`；原 nvcc 配置保留为注释便于对照。
2. **宏冲突 shim（新增 `corex_ivcore_shim.h`，force-include）**
   - `-x ivcore` 模式下 CoreX `host_defines.h` 定义了对象宏 `__noinline__ = __attribute__((noinline))`，与 Nim `nimbase.h` 的 `N_NOINLINE(rettype, name) = rettype __attribute__((__noinline__)) name` 冲突：内层 `__noinline__` 被二次展开成 `__attribute__((__attribute__((noinline))))`，导致每个翻译单元 `use of undeclared identifier 'noinline'`；
   - 修复：逐 TU `#undef __noinline__`（`__attribute__((__noinline__))` 本身是合法拼写，去掉宏遮蔽即可；`__forceinline__` 保留，因为 kernel 用到 `__device__ __forceinline__`）。
3. **nimcuda CUDA 路径重定向**
   - nimcuda 默认探测 `/usr/local/cuda`（不存在→编译期报错），用 `-d:CudaIncludes=/usr/local/corex/include`、`-d:CudaLib=/usr/local/corex/lib64`（写入 `nim.cfg`）重定向到 CoreX SDK。
4. **链接补 `-lcudart`**
   - `<<<>>>` kernel 启动生成的 `ixLaunchKernel` / `__ixPushCallConfiguration` / `__ixRegisterFunction` 等属 CoreX 运行时符号，在 `libcudart` 中；cuBLAS/cudart 的 API 调用则由 nimcuda `dlopen` 解析，故仅需 `passL: -lcudart`。
5. **cuBLAS → CoreX ixBLAS**：保留上游 `cublas*` API，运行期 dlopen `libcublas.so.10`（CoreX 的 ixBLAS 以 cublas 兼容接口导出，`nm -D` 可见 `cublasSgemm/Dgemm` 等）。float32 例程全部可用。
6. **float64 精度降级处理（按 SOP 第 3 条）**：见下节 Failure Gate。

## 结果
- **编译**：核心 CUDA 张量后端（cuBLAS + 自定义 kernel）编译 + 链接全部成功，产物 `corex_port/build/tests_cuda`。
- **测试**：上机运行 `tests/tests_cuda.nim`（5 个 CUDA 张量后端套件）——
  - 合计 36 用例：**33 通过 / 0 失败 / 3 跳过**（`tests_run = 33 + 0 + 3`）；
  - 通过的 33 项覆盖 float32 GEMM/GEMV、element-wise 加减乘除、切片/索引、reshape、broadcasting、in-place 运算、标量运算等；
  - 跳过的 3 项为纯 float64 用例（见 Failure Gate）。
- **整体**：`compile_status=success`、`test_status=partial_pass`、`overall_status=partial`。float32 CUDA 后端在 ivcore11 上功能完整、结果与 CPU 参考精确一致。

## Failure Gate（失败复现 → 分类 → 处置）
1. **float64 — terminal（硬件无真 FP64）**
   - 复现：首轮 11 个用例失败。裸探针 `corex_port/test/diag_fp64.log` 实测同一运算 float32 精确、float64 有 ~1e-7~1e-6 误差（`1.0+4.0→5.0000007`、`2.0/2→1.0000001`）；`cublasDgemm/Dgemv/Ddot` 抛 `CUBLAS_STATUS_NOT_SUPPORTED (15)`，而 `cublasS*` 全部正常。
   - 分类：ivcore11 无 FP64 运算单元，属硬件代差 **terminal**，环境内不可解。
   - 处置：按 SOP“精度降级即注释/跳过 double 用例”——
     - 纯 float64 用例（GEMV、Scalar/dot product、Addition-Substraction slices）用 `unittest.skip()` 标记为 **SKIPPED**（合理跳过，legitimate=true）；
     - float32/float64 混合用例（GEMM、若干 in-place/标量/broadcasting）仅用 `when false` 禁用 float64 断言，**保留并验证 float32 路径**（这些用例改后正常通过）。
   - 纯数据搬运的 float64 用例（Clone、Unsafe reshape，无算术）本就通过，未改动。
2. **cuDNN (-d:cudnn) — workaround-able，未纳入范围**
   - 复现：`-d:cudnn` 编译报 `cannot open file: nimcuda/cudnn`。
   - 尝试：把 `import nimcuda/cudnn` 改为 `nimcuda/cuda8_0/cudnn` → 撞类型系统不一致（`cuda12_5` 的 `check()` 无 `cudnnStatus_t` 重载；`cuda8_0` 与 `cuda12_5` 的 `cudaStream_t`/`driver_types` 不同，需为 cuDNN 单独铺 check 与流类型桥接）。
   - 判定：属上游 Arraymancer 与 nimcuda 的版本漂移（非 CoreX 特有），需跨模块较大改动，且即便打通仍会撞 float64 conv / cuDNN v7 backward `NOT_SUPPORTED`；超出本次“CUDA 张量后端”迁移范围，**已还原 `cudnn.nim` 源码保持洁净**，记入 `blockers.json`。

## 复现方式
```bash
source /etc/profile.d/corex.sh && source /root/.config/corex-migration/secrets.env
export PATH=/root/.nimble/bin:$PATH
cd /home/repos/Arraymancer
nim cpp -d:cuda -d:ssl -o:corex_port/build/tests_cuda tests/tests_cuda.nim   # 编译
./corex_port/build/tests_cuda                                                # 上机运行
```
