from .generation.backbone import PixelMeanFlowSequenceBackbone
from .generation.flow import PolaSTFlow

def build_model(args) -> PolaSTFlow:
    backbone = PixelMeanFlowSequenceBackbone(img_size=args.img_size, patch_size=args.patch_size, radar_channels=args.radar_channels, history_frames=args.history_frames, future_frames=args.future_frames, hidden_size=args.hidden_size, depth=args.prediction_depth, aux_head_depth=args.pmf_aux_head_depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, bottleneck_dim=args.bottleneck_dim, attn_drop=args.attn_drop, proj_drop=args.proj_drop, num_interval_tokens=args.pmf_interval_tokens, history_temporal_depth=args.history_temporal_depth, condition_bottleneck_dim=args.condition_bottleneck_dim, condition_mlp_ratio=args.condition_mlp_ratio)
    return PolaSTFlow(backbone=backbone, time_logit_mean=args.time_logit_mean, time_logit_std=args.time_logit_std, noise_scale=args.noise_scale, data_proportion=args.pmf_data_proportion, velocity_eps=args.velocity_eps, norm_p=args.pmf_norm_p, norm_eps=args.pmf_norm_eps, aux_v_weight=args.pmf_aux_v_weight)
