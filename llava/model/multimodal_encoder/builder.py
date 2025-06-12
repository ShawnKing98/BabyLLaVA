import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .resnext_encoder import ResNeXtVisionTower
from .vit_encoder import ViTVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if vision_tower.startswith("openai/") or vision_tower.startswith("laion/") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "dino_sfp_resnext50" in vision_tower or vision_tower.startswith("wkvong/"):
        return ResNeXtVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "dino_say_vitl16" in vision_tower:
        return ViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
