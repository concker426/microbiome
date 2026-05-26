import argparse
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 将当前目录加入路径，确保能 import 刚才写的工具类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from microbiome_inference_utils import MicrobiomeDataLoader, MicrobiomeInputConstructor

# ================= 模型配置 =================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/hd/liujx/microbiome_llm_project/saved_models/ibd_natural_lang_v2/final"
EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584
NUM_SPECIES = 6374

class MicrobiomeEncoder(nn.Module):
    """模仿 ProCyon 的 Modality-specific Encoder"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, output_dim))
    def forward(self, x): return self.net(x)

def load_model_and_adapters(dataset_type="study"):
    """模仿 ProCyon 的模型初始化流程"""
    print("🚀 [Model] 正在加载 Qwen2.5-7B 基座与 LoRA 适配器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    llm = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # 根据数据集类型设置输入维度
    if dataset_type == "ibd":
        input_dim = 300  # ibd_counts.tsv 的特征数
        print(f"📊 [Model] 使用IBD数据集配置 (输入维度: {input_dim})")
    else:  # study
        input_dim = 6374  # study_16496_counts.tsv 的特征数
        print(f"📊 [Model] 使用Study数据集配置 (输入维度: {input_dim})")
    
    # 加载自定义投影层 (ProCyon 的 connector)
    custom_layers_path = os.path.join(ADAPTER_PATH, "custom_layers.pt")
    custom_layers = torch.load(custom_layers_path, map_location="cuda:0")
    
    micro_encoder = MicrobiomeEncoder(input_dim, EMBED_DIM).to("cuda:0")
    projection = nn.Linear(EMBED_DIM, LLM_HIDDEN_SIZE).to("cuda:0")
    
    # 只有当维度匹配时才加载预训练权重
    if input_dim == 6374:  # 当前adapter是为6374维训练的
        micro_encoder.load_state_dict(custom_layers['micro_encoder'])
        projection.load_state_dict(custom_layers['projection'])
        print("✅ [Model] 加载预训练adapter权重")
    else:
        print("⚠️ [Model] 维度不匹配，使用随机初始化权重")
    
    return llm.eval(), micro_encoder.eval(), projection.eval(), tokenizer

def run_multimodal_inference(llm, micro_encoder, projection, tokenizer, input_dict):
    """
    模仿 ProCyon 的 forward + generate 逻辑。
    核心：Embedding 融合 -> Position ID 修正 -> 自回归生成
    """
    vector = input_dict["data"]["seq"]
    input_ids = input_dict["input_ids"]
    
    with torch.no_grad():
        # 1. 编码微生物特征 (Modality Encoding)
        micro_embeds = micro_encoder(vector)
        micro_tokens = projection(micro_embeds).unsqueeze(1).to(dtype=torch.float16)
        
        # 2. 文本编码 (Text Embedding)
        inputs_embeds = llm.base_model.model.model.embed_tokens(input_ids).to(dtype=torch.float16)
        
        # 3. 多模态融合 (Multimodal Fusion)
        combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1).to(dtype=torch.float16)
        
        # 4. 位置编码修复 (RoPE Alignment)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len + 1, dtype=torch.long, device="cuda:0").unsqueeze(0)
        
        # 5. 预填充 (Prefill)
        outputs = llm.model(inputs_embeds=combined_embeds, position_ids=position_ids, use_cache=True)
        
        # 6. 解码 (Decoding)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token
        current_ids = torch.cat([input_ids, next_token], dim=1)
        past_key_values = outputs.past_key_values
        
        for _ in range(128):
            pos_id = torch.full((1, 1), fill_value=current_ids.shape[1]-1, dtype=torch.long, device="cuda:0")
            out = llm.model(input_ids=next_token, position_ids=pos_id, past_key_values=past_key_values, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id: break
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            past_key_values = out.past_key_values
            
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微生物组多模态诊断系统")
    parser.add_argument("--sample_id", type=str, help="要推理的样本 ID，如果未指定则进入交互模式")
    args = parser.parse_args()

    # 初始化所有组件
    llm, micro_encoder, projection, tokenizer = load_model_and_adapters()
    data_loader = MicrobiomeDataLoader()
    input_constructor = MicrobiomeInputConstructor(tokenizer)
    
    print("\n🦠 [System] 微生物组多模态诊断系统 (ProCyon Architecture) 已启动")
    print(f"💡 [Tip] 尝试输入样本 ID: {data_loader.otu_df.index[0]}")

    def handle_sample(sample_id: str):
        vector = data_loader.get_sample_vector(sample_id)
        if vector is None:
            print(f"❌ [Error] 样本 '{sample_id}' 未在数据库中找到。")
            return
        input_dict = input_constructor.create_diagnosis_input(sample_id, vector)
        print(f"⚙️  [Inference] 正在分析样本 {sample_id} ...")
        response = run_multimodal_inference(llm, micro_encoder, projection, tokenizer, input_dict)
        print(f"\n🤖 [Assistant] {response}")

    if args.sample_id:
        handle_sample(args.sample_id)
    else:
        while True:
            user_input = input("\n👤 [User] 请输入样本 ID (quit to exit): ").strip()
            if user_input.lower() == 'quit':
                break
            handle_sample(user_input)
