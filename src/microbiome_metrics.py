import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from typing import List, Dict

class MicrobiomeEvaluator:
    """
    模仿 ProCyon 的评估逻辑，提供分类与生成质量的多维度衡量。
    """
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []
        self.losses = []

    def update_classification(self, true_label: str, pred_text: str):
        """
        解析模型输出并更新分类指标。
        支持二分类和四分类标签。
        """
        self.true_labels.append(true_label)
        
        # 智能关键词提取逻辑，支持多分类
        pred_text_upper = pred_text.upper()
        pred_label = "Unknown"
        
        # 优先匹配更具体的标签
        if "CROHN" in pred_text_upper or "CD" in pred_text_upper:
            pred_label = "CD"
        elif "ULCERATIVE" in pred_text_upper or "UC" in pred_text_upper:
            pred_label = "UC"
        elif "IBD" in pred_text_upper and pred_label == "Unknown":  # 只有在没匹配到CD/UC时才设为IBD
            pred_label = "IBD"
        elif "HEALTHY" in pred_text_upper or "NORMAL" in pred_text_upper:
            pred_label = "Healthy"
        
        self.pred_labels.append(pred_label)

    def update_loss(self, loss: float):
        """记录训练/推理过程中的 Loss"""
        self.losses.append(loss)

    def compute_metrics(self) -> Dict:
        """
        计算最终的 Accuracy, Precision, Recall, F1-Score。
        支持二分类和多分类评估。
        """
        if not self.true_labels:
            return {}

        # 过滤掉无法识别的预测
        valid_pairs = [(t, p) for t, p in zip(self.true_labels, self.pred_labels) if p != "Unknown"]
        if not valid_pairs:
            return {"error": "No valid predictions found"}

        y_true, y_pred = zip(*valid_pairs)
        unique_labels = set(y_true)
        
        # 多分类评估
        if len(unique_labels) > 2:
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
            
            # 各类别详细指标
            detailed_metrics = {}
            for label in sorted(unique_labels):
                if label in y_pred or label in y_true:
                    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[label], average=None)
                    detailed_metrics[f"{label}_precision"] = round(p[0], 4)
                    detailed_metrics[f"{label}_recall"] = round(r[0], 4)
                    detailed_metrics[f"{label}_f1"] = round(f[0], 4)
                    detailed_metrics[f"{label}_support"] = s[0]
            
            return {
                "Accuracy": round(acc, 4),
                "Macro_Precision": round(precision, 4),
                "Macro_Recall": round(recall, 4),
                "Macro_F1": round(f1, 4),
                "Total_Samples": len(valid_pairs),
                **detailed_metrics
            }
        else:
            # 二分类评估（兼容旧版本）
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label='IBD')
            
            return {
                "Accuracy": round(acc, 4),
                "Precision (IBD)": round(precision, 4),
                "Recall (IBD)": round(recall, 4),
                "F1-Score (IBD)": round(f1, 4),
                "Total_Samples": len(valid_pairs)
            }

    def compute_perplexity(self) -> float:
        """根据平均 Loss 计算困惑度 (PPL)"""
        if not self.losses:
            return 0.0
        avg_loss = np.mean(self.losses)
        return np.exp(avg_loss)
