from pathlib import Path

student_revision = "8f928dd73899dd9279a878d732841919a98e61e6"
student_revision_url = (
    "https://huggingface.co/musetric/mel-band-roformer-duality-kara2-student/resolve/"
    f"{student_revision}"
)
student_weights_url = f"{student_revision_url}/model.safetensors"
student_config_url = f"{student_revision_url}/config.yaml"
student_dir = Path("mel_band_roformer_duality_kara2_student") / student_revision
student_weights_rel_path = student_dir / "model.safetensors"
student_config_rel_path = student_dir / "config.yaml"

rmvpe_checkpoint_url = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
    "e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt"
)
rmvpe_dir = "rmvpe"
rmvpe_checkpoint_rel_path = Path(rmvpe_dir) / "rmvpe.pt"
