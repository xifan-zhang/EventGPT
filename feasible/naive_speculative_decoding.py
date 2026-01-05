import torch
from transformers import AutoModelForCausalLM
from tokenizer_check import load_eventgpt_tokenizer

EVENTGPT_MODEL_PATH = "./checkpoints/EventGPT-7b"

class NaiveSpeculativeDecoder:
    def __init__(self, draft_model, target_model, tokenizer):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer

    def generate(self, input_ids, max_new_tokens=50, num_draft_tokens=5):
        """
        最简单的speculative decoding实现
        """
        generated = input_ids.clone()
        acceptance_history = []

        for _ in range(max_new_tokens // num_draft_tokens):
            # Phase 1: Draft model生成K个tokens
            draft_outputs = []
            current_ids = generated

            for k in range(num_draft_tokens):
                with torch.no_grad():
                    draft_logits = self.draft(current_ids).logits[:, -1, :]
                    draft_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
                    draft_outputs.append(draft_token)
                    current_ids = torch.cat([current_ids, draft_token], dim=1)

            # Phase 2: Target model并行验证所有draft tokens
            with torch.no_grad():
                target_logits = self.target(current_ids).logits

            # Phase 3: 逐token验证接受/拒绝
            accepted_count = 0
            for k in range(num_draft_tokens):
                draft_token = draft_outputs[k]
                target_probs = torch.softmax(
                    target_logits[:, -(num_draft_tokens-k), :], dim=-1
                )

                # 简化的接受准则：greedy decoding
                target_best = torch.argmax(target_probs, dim=-1, keepdim=True)

                if draft_token.item() == target_best.item():
                    accepted_count += 1
                    generated = torch.cat([generated, draft_token], dim=1)
                else:
                    # 拒绝，使用target的预测
                    generated = torch.cat([generated, target_best], dim=1)
                    break

            acceptance_history.append(accepted_count / num_draft_tokens)

            if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break

        return generated, acceptance_history

if __name__ == "__main__":
    eventgpt_tokenizer = load_eventgpt_tokenizer(EVENTGPT_MODEL_PATH)

    # 使用示例
    draft_model = AutoModelForCausalLM.from_pretrained(EVENTGPT_MODEL_PATH)
    target_model = AutoModelForCausalLM.from_pretrained("llava-hf/llava-1.5-13b-hf")

    decoder = NaiveSpeculativeDecoder(draft_model, target_model, eventgpt_tokenizer)
