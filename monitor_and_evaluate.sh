#!/bin/bash

echo "🔍 开始监控训练进程..."

# 等待训练完成
while ps aux | grep -q "[t]rain_merged_stable.py"; do
    sleep 30
    echo "⏳ 训练仍在进行中... ($(date '+%H:%M:%S'))"
done

echo ""
echo "✅ 训练已完成！"
echo ""

# 检查模型是否保存
if [ -d "saved_models/merged_multidataset_v3/final" ]; then
    echo "📦 检测到新模型: saved_models/merged_multidataset_v3/final/"
    echo ""
    
    # 运行评估
    echo "🧪 开始使用测试集评估新模型..."
    python3 src/evaluate_with_real_data.py
    
    echo ""
    echo "📊 评估完成！查看结果:"
    echo "   cat evaluation_results_real/evaluation_report.txt"
else
    echo "❌ 未找到模型文件"
fi
