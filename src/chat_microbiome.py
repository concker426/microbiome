import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch.nn as nn

# ================= 配置区域 =================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/hd/liujx/microbiome_llm_project/saved_models/ibd_natural_lang_v2/final"
COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
NUM_SPECIES = 6374 
EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584

class MicrobiomeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, output_dim))
    def forward(self, x): return self.net(x)

def load_model_and_data():
    print("🔄 正在加载模型和权重...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map={"": "cuda:0"}, local_files_only=True)
    llm = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    custom_layers = torch.load(os.path.join(ADAPTER_PATH, "custom_layers.pt"), map_location="cuda:0")
    micro_encoder = MicrobiomeEncoder(NUM_SPECIES, EMBED_DIM).to("cuda:0")
    projection = nn.Linear(EMBED_DIM, LLM_HIDDEN_SIZE).to("cuda:0")
    micro_encoder.load_state_dict(custom_layers['micro_encoder'])
    projection.load_state_dict(custom_layers['projection'])
    
    df = pd.read_csv(COUNTS_FILE, sep='\t', index_col=0, skiprows=2, low_memory=False, dtype=str)
    if df.shape[1] == NUM_SPECIES:
        pass
    else:
        df = df.T
        
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"✅ 数据加载成功！形状: {df.shape}")
    print(f"💡 示例 ID: {df.index[0]}")
    
    return llm.eval(), micro_encoder.eval(), projection.eval(), tokenizer, df

def chat(llm, micro_encoder, projection, tokenizer, otu_df):
    print("\n🦠 微生物组专家 AI 已就绪！(输入 'quit' 退出)")
    
    while True:
        user_input = input("\n👤 请输入样本 ID: ").strip()
        if user_input.lower() == 'quit': break
        
        matched_id = None
        if user_input in otu_df.index:
            matched_id = user_input
        else:
            for idx in otu_df.index:
                if user_input in idx:
                    matched_id = idx
                    break

        if not matched_id:
            print(f"❌ 找不到样本。请尝试输入: {otu_df.index[0]}")
            continue
            
        print(f"✅ 正在分析样本: {matched_id} ...")
        otu_vector_raw = otu_df.loc[matched_id].values
        
        if len(otu_vector_raw) != NUM_SPECIES:
            print(f"⚠️ 维度错误: {len(otu_vector_raw)} vs {NUM_SPECIES}")
            continue
            
        otu_vector = torch.tensor(otu_vector_raw.astype(np.float32)).unsqueeze(0).to("cuda:0")
        
        prompt = f"你是一位专业的肠道微生物分析师。请分析样本 {matched_id} 的菌群数据。\n\n【主要菌群构成】: 请参考提供的数值向量。\n\n请判断该样本的健康状态（Healthy 或 IBD），并简要说明理由。"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            # 1. 编码微生物数据
            micro_embeds = micro_encoder(otu_vector)
            micro_tokens = projection(micro_embeds).unsqueeze(1).to(dtype=torch.bfloat16)
            
            # 2. 获取文本 Embeddings
            inputs_embeds = llm.base_model.model.model.embed_tokens(inputs.input_ids)
            
            # 3. 融合
            combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
            
            # 4. 【关键修复】手动构造 position_ids
            # 原本有 seq_len 个位置，现在多了一个，所以是 0 到 seq_len
            position_ids = torch.arange(0, seq_len + 1, dtype=torch.long, device="cuda:0").unsqueeze(0)
            
            # 5. 生成回答 (使用 model 而不是 generate，以便更精细控制)
            # 这里我们先用 model 跑一次 prefill，拿到 past_key_values，然后再 decode
            outputs = llm.model(
                inputs_embeds=combined_embeds,
                position_ids=position_ids,
                use_cache=True
            )
            
            # 简单的 Greedy Search 解码第一步
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 继续生成剩余部分
            generated_ids = next_token
            current_ids = torch.cat([inputs.input_ids, next_token], dim=1)
            past_key_values = outputs.past_key_values
            
            for _ in range(50): # 限制生成长度
                pos_id = torch.full((1, 1), fill_value=current_ids.shape[1]-1, dtype=torch.long, device="cuda:0")
                out = llm.model(
                    input_ids=next_token,
                    position_ids=pos_id,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                past_key_values = out.past_key_values

            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
        print(f"\n🤖 AI 诊断报告:\n{response}")

if __name__ == "__main__":
    try:
        llm, micro_encoder, projection, tokenizer, otu_df = load_model_and_data()
        chat(llm, micro_encoder, projection, tokenizer, otu_df)
    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        traceback.print_exc()
