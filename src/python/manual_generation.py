import torch
from transformers import GPT2Tokenizer
import sys
sys.path.append('src/python')
from custom_gpt2 import create_flash_gpt2

def manual_generate(model, input_ids, max_length=50):
    generated = input_ids.clone()

    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == 50256:
            break

    return generated

def test_manual_generation():
    print("="*70)
    print("Manual Generation Test")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create Flash model
    print("\nLoading Flash GPT-2...")
    flash_model = create_flash_gpt2("gpt2", use_flash=True, replace_layers=None).cuda().eval()

    # Test prompt
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Input shape: {inputs.input_ids.shape}")

    # Manual generation
    print("\nGenerating with Flash GPT-2 (manual loop)...")
    generated = manual_generate(flash_model, inputs.input_ids, max_length=50)

    text = tokenizer.decode(generated[0])
    print(f"\nGenerated text:")
    print(text)

    print("\n" + "="*70)
    print("✓ Manual generation complete!")

if __name__ == "__main__":
    test_manual_generation()
