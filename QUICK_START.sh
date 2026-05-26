#!/bin/bash
# 多数据集训练快速启动脚本

echo "=========================================="
echo "微生物组LLM - 多数据集联合训练"
echo "=========================================="
echo ""

# 检查必要文件
echo "📋 检查必要文件..."
if [ ! -f "data/study_16496_counts.tsv" ]; then
    echo "❌ 错误: 找不到 study_16496_counts.tsv"
    exit 1
fi

if [ ! -f "data/ibd_counts.tsv" ]; then
    echo "❌ 错误: 找不到 ibd_counts.tsv"
    exit 1
fi

echo "✅ 数据文件存在"
echo ""

# 步骤1: 生成合并数据
echo "=========================================="
echo "步骤 1/2: 生成合并训练数据"
echo "=========================================="
python3 src/merge_datasets.py

if [ $? -ne 0 ]; then
    echo "❌ 数据合并失败"
    exit 1
fi

echo ""
echo "✅ 数据合并完成"
echo ""

# 询问是否开始训练
read -p "是否现在开始训练？(y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=========================================="
    echo "步骤 2/2: 开始训练"
    echo "=========================================="
    echo ""
    echo "⚠️  预计训练时间: 6-12小时"
    echo "⚠️  显存需求: 至少24GB GPU"
    echo ""
    
    python3 src/train_merged_multidataset.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "🎉 训练完成！"
        echo "=========================================="
        echo "模型保存在: saved_models/merged_multidataset_v1/"
        echo ""
        echo "下一步:"
        echo "  1. 查看训练日志和loss曲线"
        echo "  2. 使用评估脚本测试模型性能"
        echo "  3. 如果效果不理想，调整超参数重新训练"
    else
        echo ""
        echo "❌ 训练失败，请检查错误信息"
        exit 1
    fi
else
    echo ""
    echo "已跳过训练步骤"
    echo ""
    echo "要开始训练，请运行:"
    echo "  python3 src/train_merged_multidataset.py"
fi

echo ""
echo "详细说明请查看: MULTI_DATASET_TRAINING_GUIDE.md"
