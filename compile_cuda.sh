#!/bin/bash

mkdir -p cache

if [ ! -d "cuda" ]; then
    echo "❌ 错误: 未找到 'cuda' 文件夹"
    exit 1
fi

echo "开始遍历编译 CUDA 文件..."

for file in cuda/*.cu; do
    filename=$(basename "$file" .cu)
    echo "正在编译: $file -> cache/$filename"
    
    nvcc "$file" -o "cache/$filename"
    
    if [ $? -eq 0 ]; then
        echo "✅ 编译成功: cache/$filename"
    else
        echo "❌ 编译失败: $file"
    fi
done

echo "所有任务处理完毕！"