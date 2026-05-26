import argparse
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# 将当前目录加入路径，确保能 import 刚才写的工具类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from microbiome_inference_utils import MicrobiomeDataLoader, MicrobiomeInputConstructor

# ================= 模型配置 =================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584
NUM_SPECIES = 6374

class MicrobiomeEncoder(nn.Module):
    """模仿 ProCyon 的 Modality-specific Encoder"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, output_dim))
    def forward(self, x): return self.net(x)

def load_base_model():
    """加载 Qwen2.5-7B 基座模型（零样本，无 LoRA）"""
    print("🚀 [Model] 正在加载 Qwen2.5-7B 基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    
    # 加载自定义投影层 (ProCyon 的 connector)
    custom_layers_path = "/hd/liujx/microbiome_llm_project/saved_models/ibd_natural_lang_v2/final/custom_layers.pt"
    custom_layers = torch.load(custom_layers_path, map_location="cuda:0")
    
    micro_encoder = MicrobiomeEncoder(NUM_SPECIES, EMBED_DIM).to("cuda:0").half()
    projection = nn.Linear(EMBED_DIM, LLM_HIDDEN_SIZE).to("cuda:0").half()
    
    micro_encoder.load_state_dict(custom_layers['micro_encoder'])
    projection.load_state_dict(custom_layers['projection'])
    
    return base_model, micro_encoder, projection, tokenizer

def run_zero_shot_inference(llm, micro_encoder, projection, tokenizer, sample_id, data_loader):
    """零样本推理：不加载 LoRA，直接用基座模型"""
    vector = data_loader.get_sample_vector(sample_id)
    if vector is None:
        return f"❌ 样本 {sample_id} 未找到"
    
    # 构造输入
    input_constructor = MicrobiomeInputConstructor(tokenizer)
    input_dict = input_constructor.create_diagnosis_input(sample_id, vector)
    
    # 多模态融合
    text_tokens = input_dict['input_ids'].to("cuda:0")
    micro_tokens = torch.as_tensor(vector, dtype=torch.float16, device="cuda:0")
    micro_embeds = micro_encoder(micro_tokens)
    micro_tokens = projection(micro_embeds).unsqueeze(1)

    inputs_embeds = llm.get_input_embeddings()(text_tokens)
    combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
    combined_mask = torch.cat([input_dict['attention_mask'].to("cuda:0"), torch.ones_like(micro_tokens[..., 0])], dim=1)

    # 生成
    with torch.no_grad():
        outputs = llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Baseline Testing: Zero-shot Inference with Qwen2.5-7B")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    args = parser.parse_args()
    
    print("🦠 [System] 微生物组零样本诊断系统 (Baseline Testing) 已启动")
    
    # 初始化组件
    llm, micro_encoder, projection, tokenizer = load_base_model()
    data_loader = MicrobiomeDataLoader()
    
    # 选择前 N 个样本进行测试
    sample_ids = data_loader.otu_df.index.tolist()[:args.num_samples]
    
    print(f"📊 [Info] 开始对 {len(sample_ids)} 个样本进行零样本推理...")
    
    for sid in sample_ids:
        print(f"\n⚙️  [Inference] 正在分析样本 {sid} ...")
        response = run_zero_shot_inference(llm, micro_encoder, projection, tokenizer, sid, data_loader)
        print(f"🤖 [Assistant] 零样本诊断结果: {response}")
    
    print("\n✅ [Baseline Testing] 完成。观察模型是否能理解微生物数据并生成通顺诊断。")

if __name__ == "__main__":
    main()