from pathlib import Path

# Third-party model/data sources used by the audio workflows.
# See thirdPartyNotices.md for source, license, and attribution details.
model_checkpoint_url = "https://huggingface.co/Aname-Tommy/Mel-Band-Roformer_Duality/resolve/07a189ffdceb69db0ddacc5175569bef59850016/duality_v1.ckpt"
model_config_url = "https://huggingface.co/Aname-Tommy/Mel-Band-Roformer_Duality/resolve/07a189ffdceb69db0ddacc5175569bef59850016/config_v1.yaml"

model_mel_band_roformer_dir = "mel_band_roformer_duality"
model_checkpoint_rel_path = Path(model_mel_band_roformer_dir) / "model.ckpt"
model_config_rel_path = Path(model_mel_band_roformer_dir) / "config.yaml"

# The vocals model of the product: the student distilled from Duality.
student_revision_url = (
    "https://huggingface.co/musetric/mel-band-roformer-duality-student/resolve/"
    "c9d2c65a65d820bdd4efdc7d62bb348998969592"
)
student_weights_url = f"{student_revision_url}/model.safetensors"
student_config_url = f"{student_revision_url}/config.yaml"
student_dir = "mel_band_roformer_duality_student"
student_weights_rel_path = Path(student_dir) / "model.safetensors"
student_config_rel_path = Path(student_dir) / "config.yaml"

karaoke_mdx_model_url = (
    "https://huggingface.co/AI4future/RVC/resolve/main/UVR_MDXNET_KARA_2.onnx"
)
karaoke_mdx_models_dir = "uvr_mdxnet_kara_2"
karaoke_mdx_model_rel_path = Path(karaoke_mdx_models_dir) / "UVR_MDXNET_KARA_2.onnx"

mdx_model_data_url = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
    "mdx_model_data/model_data_new.json"
)
mdx_model_data_dir = "mdx_model_data"
mdx_model_data_rel_path = Path(mdx_model_data_dir) / "model_data_new.json"

rmvpe_checkpoint_url = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
    "e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt"
)
rmvpe_dir = "rmvpe"
rmvpe_checkpoint_rel_path = Path(rmvpe_dir) / "rmvpe.pt"
