from src.utils.model_utils import load_model

model, tokenizer = load_model("Qwen/Qwen3-8B")

tok = tokenizer

# im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
# print("im_start_id:", im_start_id)
#
# # sanity check: decode the id
# print("decoded:", tok.decode([im_start_id], skip_special_tokens=False))
#
# if im_start_id in (None, -1) or im_start_id == tok.unk_token_id:
#     raise ValueError("'<|im_start|>' is not a standalone token in this tokenizer vocab.")

print("bos_token:", tok.bos_token)
print("bos_token_id:", tok.bos_token_id)

print("eos_token:", tok.eos_token)
print("eos_token_id:", tok.eos_token_id)

print("special_tokens_map:", tok.special_tokens_map)
print("all_special_tokens:", tok.all_special_tokens)
print("all_special_ids:", tok.all_special_ids)

special_tokens_map = {
    'eos_token': '<|im_end|>',
    'pad_token': '<|im_end|>',
    'additional_special_tokens': [
    '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>',
    '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>',
    '<|video_pad|>']}
