# import os
# from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_cfg, 's2', False)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')
import torch
import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .vit_3d import ViT

def build_vision_tower(vision_tower_cfg, **kwargs):
    '''
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')
    '''
    vit = ViT(
            image_size = 256,          # image size
            frames = 512,               # max number of frames
            image_patch_size = 32,     # image patch size
            frame_patch_size = 4,      # frame patch size
            dim = 768,
            depth = 12,
            heads = 8,
            mlp_dim = 2048,
            channels=1,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    
    vit.hidden_size = 49152

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    if vision_tower is not None:
        is_absolute_path_exists = os.path.exists(vision_tower)
    
        if is_absolute_path_exists:
            print("Loading Vision Tower from: ", vision_tower)
            checkpoint = torch.load(vision_tower, map_location='cpu')
            vit.load_state_dict(checkpoint, strict=False) 
            
            for param in vit.parameters():
                param.requires_grad = False

            vit.eval() 
            
            vit.is_loaded = True
        
            return vit
        
        return None
    
    return None 
    
