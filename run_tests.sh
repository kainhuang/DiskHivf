#!/bin/bash
#
# DiskHIVF 异步 I/O 优化 — 一键编译并运行全部单元测试
#
# 使用方式:
#   chmod +x run_tests.sh
#   ./run_tests.sh
#
# 前提条件:
#   - gcc 4.8+ 且支持 -std=c++11
#   - 在项目根目录下运行，或者在 src 目录下运行
#

set -e

# 定位到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/makefile" ] || [ -f "$SCRIPT_DIR/Makefile" ]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

SRC_DIR="$PROJECT_ROOT/src"
BIN_DIR="$PROJECT_ROOT/bin"

echo "========================================"
echo "DiskHIVF 异步 I/O 优化 — 一键测试"
echo "========================================"
echo ""
echo "项目目录: $PROJECT_ROOT"
echo "源码目录: $SRC_DIR"
echo "输出目录: $BIN_DIR"
echo ""

# 确保 bin 目录存在
mkdir -p "$BIN_DIR"

# 编译
echo "--- 编译中 ---"
cd "$SRC_DIR"

# 先编译依赖的 .o 文件
echo "  编译 unity.o ..."
make unity.o 2>&1 || true
echo "  编译 random.o ..."
make random.o 2>&1 || true
echo "  编译 conf.o ..."
make conf.o 2>&1 || true
echo "  编译 matrix.o ..."
make matrix.o 2>&1 || true
echo "  编译 kmeans.o ..."
make kmeans.o 2>&1 || true
echo "  编译 file_read_write.o ..."
make file_read_write.o 2>&1 || true
echo "  编译 thread_pool.o ..."
make thread_pool.o 2>&1 || true
echo "  编译 lru_cache.o ..."
make lru_cache.o 2>&1 || true
echo "  编译 hierachical_cluster.o ..."
make hierachical_cluster.o 2>&1 || true

echo "  编译 test_async_io ..."
make test_async_io 2>&1
COMPILE_RET=$?

if [ $COMPILE_RET -ne 0 ]; then
    echo ""
    echo "❌ 编译失败！请检查上方错误信息。"
    exit 1
fi

echo ""
echo "✅ 编译成功"
echo ""

# 运行测试
echo "--- 运行单元测试 ---"
echo ""
cd "$PROJECT_ROOT"

"$BIN_DIR/test_async_io"
TEST_RET=$?

echo ""
if [ $TEST_RET -eq 0 ]; then
    echo "========================================"
    echo "✅ 所有单元测试通过！"
    echo "========================================"
else
    echo "========================================"
    echo "❌ 存在失败的测试，退出码: $TEST_RET"
    echo "========================================"
fi

exit $TEST_RET
