import json
import os


def create_preprocessor_config(image_processor, output_dir):
    """Serialize a robust preprocessor_config.json for any ImageProcessor.

    Attempts to use the processor's own serialization and fills in common
    fields from attributes or timm-style data_config to avoid hardcoding
    specific processor types.
    """
    try:
        cfg = image_processor.to_dict() if hasattr(image_processor, "to_dict") else {}

        dc = getattr(image_processor, "data_config", None)
        dc = dc if isinstance(dc, dict) else {}

        mean = cfg.get("image_mean") or getattr(image_processor, "image_mean", None) or dc.get("mean")
        std = cfg.get("image_std") or getattr(image_processor, "image_std", None) or dc.get("std")

        size_cfg = cfg.get("size") or getattr(image_processor, "size", None)
        if size_cfg is None:
            inp = dc.get("input_size")
            if isinstance(inp, dict) and "shortest_edge" in inp:
                size_cfg = {"shortest_edge": int(inp["shortest_edge"])}
            elif isinstance(inp, (list, tuple)) and len(inp) >= 3:
                # (C, H, W)
                size_cfg = {"height": int(inp[1]), "width": int(inp[2])}

        if mean is not None:
            cfg["image_mean"] = mean
        if std is not None:
            cfg["image_std"] = std
        if size_cfg is not None:
            cfg["size"] = size_cfg

        cfg.setdefault("do_resize", True if size_cfg else cfg.get("do_resize", True))
        cfg.setdefault("do_normalize", True if (mean and std) else cfg.get("do_normalize", True))
        cfg.setdefault("do_rescale", cfg.get("do_rescale", True))
        cfg.setdefault("rescale_factor", cfg.get("rescale_factor", 1 / 255))
        cfg.setdefault("resample", cfg.get("resample", 3))  # 3 = BICUBIC
        cfg.setdefault("image_processor_type", cfg.get("image_processor_type", type(image_processor).__name__))

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        # Keep training running even if serialization fails
        print(f"Error creating preprocessor_config: {e}")

