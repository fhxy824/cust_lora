import torch

# Creating a dummy state_dict
dummy_state_dict = {
    'fc1.weight': torch.randn(3, 4),  # Random tensor with shape (50, 10) for fc1's weight
    'fc1.bias': torch.randn(6)  # Random tensor with shape (5) for fc3's bias
}

BLOCKS = {
    'content': ['fc1.weight'],
    'style': ['unet.up_blocks.0.attentions.1'],
}

# Printing the dummy state_dict
for key, value in dummy_state_dict.items():
    print(f'{key}: {value.shape}')


def filter_lora(state_dict,blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')

def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')
    
re1=filter_lora(dummy_state_dict,BLOCKS['content'])
# Printing the shape of each tensor in the state_dict
print("Shapes of each tensor in dummy_state_dict:")
for key, value in re1.items():
    print(f'{key}: {value.size()}')

# Calculating the total number of elements in the state_dict
total_elements = sum(value.numel() for value in re1.values())
print(f"\nTotal number of elements in dummy_state_dict: {total_elements}")

s1='abcc'
s2='abccdex'
print(s1.startswith(s2))
print(s2.startswith(s1))