"""
Author: wzh
Since: 2023-9-10
"""
import torch

transfer_map = {'features.0':'conv1_1', 'features.2':'conv1_2',
                'features.5':'conv2_1', 'features.7':'conv2_2',
                'features.10':'conv3_1', 'features.12':'conv3_2',
                'features.14':'conv3_3', 'features.17':'conv4_1',
                'features.19':'conv4_2', 'features.21':'conv4_3',
                'features.24':'conv5_1', 'features.26':'conv5_2',
                'features.28':'conv5_3', 'features.32':'conv6_1',
                'features.35':'conv7_1', 'features.38':'conv8'}
test = torch.load("deeplab_largeFOV.pth")
new_dict = {}
for key, value in test.items():
    if 'weight' in key:
        old_name = key[:-7]
    elif 'bias' in key:
        old_name = key[:-5]
    new_key = key.replace(old_name, transfer_map[old_name])
    new_dict[new_key] = value
torch.save(new_dict, "DCAN_pretrained_weight.pth")
print('Finish')



