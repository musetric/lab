import torch

from musetric_toolkit.key_audio.skey_checkpoint import ensure_checkpoint


def run_skey(audio_path: str, models_path: str) -> tuple[str, str, float]:
    from musetric_toolkit.key_audio.skey.key_detection import (  # noqa: PLC0415
        key_map,
        load_audio,
        load_checkpoint,
        load_model_components,
    )

    checkpoint_path = ensure_checkpoint(models_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path)
    sample_rate = ckpt["audio"]["sr"]
    hcqt, chromanet, crop_fn = load_model_components(ckpt, device)

    audio = load_audio(audio_path, sample_rate).to(device)
    batch = audio.unsqueeze(0)
    with torch.no_grad():
        cropped = crop_fn(hcqt(batch), torch.zeros(1).to(device))
        logits = chromanet(cropped)
        mean_logits = torch.mean(logits, dim=0)
        probs = torch.softmax(mean_logits, dim=-1)
        idx = int(probs.argmax())
        confidence = float(probs[idx])

    key_str = key_map[idx]
    root, mode_str = key_str.split(" ", 1)
    return root, mode_str.lower(), confidence
