# Allows us to cache and alter activations
from transformer_lens import HookedTransformer
import torch
from acdc.docstring.utils import AllDataThings
import itertools
from transformers import GPT2Tokenizer
import random
import json
from acdc.acdc_utils import negative_log_probs
from functools import partial

# Get a tokeniser, think parsers - 
# 'Tokens’ are streams of characters that GPT-2 might recognise as 1 word or combination of characters
# For instance, [str, tan, tax, rge, clo, blk] may define a collection of tokens
# Whereas [cck, kta, nh] defines a smaller collection where each item is 2 tokens each
# Honestly, the classifications are rather confusing, but it applies later - only an overview is needed
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def token_count(text):
    return len(tokenizer.encode(text))

# See TransformerLens Documentation
# Retrieves a pretrained model from TransformerLens with hooks pre-attached
def get_iterate_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a pretrained GPT-2 style model
    model = HookedTransformer.from_pretrained("gpt-neo-125M", device=device)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    return model

# Regular samples have form: 'def iterator(1TokenString):'
# Where 1TokenString is a random string that comes to exactly 1 token
def generate_samples(num_samples):
    
    # Generate potential values for B
    characters = "abcdefghijklmnopqrstuvwxyz"
    valid_Bs = []

    # Generate values for B (exactly 1 token)
    for i in range(2, 4):  # Try lengths from 2 to 3 characters
        for combo in itertools.product(characters, repeat=i):
            B_candidate = "".join(combo)
            if token_count(B_candidate) == 1:
                valid_Bs.append(B_candidate)
            if len(valid_Bs) >= num_samples:
                break
        if len(valid_Bs) >= num_samples:
            break

    # Generate function definitions and save to file
    return [f'def iterator({B}):\n\t' for B in valid_Bs]

# Corrupted samples have form: 'def 2TokenString(1TokenString):'
# Where 2TokenString is a random string that comes to exactly 2tokens
def generate_corrupted_samples(num_samples):
    # Generate potential values for A and B
    characters = "abcdefghijklmnopqrstuvwxyz"

    # Create lists to store 1000 valid values for A and B
    valid_As = []
    valid_Bs = []

    # Generate values for A (exactly 2 tokens)
    for i in range(2, 6):  # Try lengths from 2 to 5 characters
        for combo in itertools.product(characters, repeat=i):
            A_candidate = "".join(combo)
            if token_count(A_candidate) == 2:
                valid_As.append(A_candidate)
            if len(valid_As) >= num_samples:
                break
        if len(valid_As) >= num_samples:
            break

    # Generate values for B (exactly 1 token)
    for i in range(2, 4):  # Try lengths from 2 to 3 characters
        for combo in itertools.product(characters, repeat=i):
            B_candidate = "".join(combo)
            if token_count(B_candidate) == 1:
                valid_Bs.append(B_candidate)
            if len(valid_Bs) >= num_samples:
                break
        if len(valid_Bs) >= num_samples:
            break

    # Generate function definitions by pairing A and B
    samples = []
    i = 0
    while i < num_samples:
        A = random.choice(valid_As)
        B = random.choice(valid_Bs)
        sample = f'def {A}({B}):\n\t'
        if token_count(sample) == 8:
            samples.append(sample)
            i += 1

    return samples


KEYWORD = 'for'

# Think back to the 'get_all_X_things' in acdc/main.py
def get_all_iterate_things(num_samples: int, device = None):

    # Generate the samples and corrupted samples for activation patching, then prepare inputs for the algorithm
    random.seed(0)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tl_model = get_iterate_model(device)
    data = generate_samples(num_samples * 2)
    patch_data = generate_corrupted_samples(num_samples * 2)
    data = torch.tensor(tokenizer([tokenizer.bos_token + d for d in data]).input_ids).to(device) # num_samples * context length
    p_d = [tokenizer.bos_token + d for d in patch_data]    
    patch_data = torch.tensor(tokenizer(p_d).input_ids).to(device) # num_samples * context length

    # Train test splits?
    val_data = data[:num_samples,:]
    test_data = data[num_samples:,:]
    val_patch_data = patch_data[:num_samples,:]
    test_patch_data = patch_data[num_samples:,:]

    # the correct label for each prompt is "for"
    '''
    def iterate(arr):
        [for]
    '''
    labels = torch.tensor(tokenizer.encode(KEYWORD) * data.shape[0]).to(device)
    val_labels = labels[:num_samples]
    test_labels = labels[num_samples:]
    #return data, patch_data

    val_metric = partial(
        negative_log_probs,
        labels = val_labels,
        last_seq_element_only=True
    )

    test_metric = partial(
        negative_log_probs,
        labels = test_labels,
        last_seq_element_only=True
    )

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=val_metric,
        validation_data=val_data,
        validation_labels=val_labels,
        validation_mask=None,
        validation_patch_data=val_patch_data,
        test_metrics=test_metric,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data
    )

def save_final_logits():
    things = get_all_iterate_things(1000)
    torch.save(things.validation_data, 'val_dataset.pt')
    torch.save(things.validation_patch_data, 'val_patch_dataset.pt')

if __name__ == '__main__':
    save_final_logits()