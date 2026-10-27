# Third-party Notices

This project includes or adapts portions of the following third-party works.
This notice focuses on vendored code, adapted implementations, model/data
artifacts downloaded by the lab, and primary runtime solutions used by the
CLI workflows. Ordinary Python package dependencies keep their license metadata
in their distributed packages.

## Ultimate Vocal Remover GUI / MDX-Net

- Source: https://github.com/Anjok07/ultimatevocalremovergui
- Usage: MDX-Net ONNX inference flow, model tensor layout, chunked demix / overlap-add behavior, and compatibility with UVR-trained MDX models.
- Local files: `musetric_toolkit/separate_audio/mdx_net_separator.py`, `musetric_toolkit/separate_audio/main.py`.
- License: MIT.
- License source: upstream README; no repository license file was found when this notice was prepared.
- Credit: Ultimate Vocal Remover GUI / UVR developers, including Anjok07 and aufr33; original MDX-Net AI code credited upstream to Kuielab and Woosung Choi.

The upstream repository README asks third-party application developers who use UVR models to credit UVR and its developers.

## AI4future/RVC - UVR_MDXNET_KARA_2.onnx

- Source: https://huggingface.co/AI4future/RVC
- Usage: `UVR_MDXNET_KARA_2.onnx` model downloaded at runtime for lead/backing vocal separation.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/main.py`.
- License: MIT.
- License source: Hugging Face model card metadata.

## TRvlvr/application_data

- Source: https://github.com/TRvlvr/application_data
- Usage: MDX model hash-to-parameter metadata (`mdx_model_data/model_data_new.json`) used to configure UVR-compatible MDX models.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/mdx_net_separator.py`.
- License: no repository license file was found when this notice was prepared.

## DeepExtract

- Source: https://github.com/abdozmantar/deepextract
- Usage: STFT / inverse-STFT tensor conversion pattern used by the MDX ONNX separator.
- Local files: `musetric_toolkit/separate_audio/mdx_net_separator.py`.
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) [2024] Abdullah Ozmantar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BS-RoFormer

- Source: https://github.com/lucidrains/BS-RoFormer
- Usage: `MelBandRoformer` and attention model architecture adapted for local separation checkpoints.
- Local files: `musetric_toolkit/separate_audio/roformer/mel_band_roformer.py`, `musetric_toolkit/separate_audio/roformer/attend.py`, `musetric_toolkit/separate_audio/roformer/pooled_mel_band_roformer.py` (built on the first).
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Aname-Tommy/Mel-Band-Roformer_Duality

- Source: https://huggingface.co/Aname-Tommy/Mel-Band-Roformer_Duality, revision `07a189ffdceb69db0ddacc5175569bef59850016`.
- Usage: checkpoint (`duality_v1.ckpt`) and config (`config_v1.yaml`) downloaded at runtime for vocal/instrumental separation, and the source of the ONNX core exported by `scripts/onnx/roformer`.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/main.py`, `musetric_toolkit/separate_audio/mel_band_roformer_separator.py`, `scripts/onnx/roformer/converting.md`.
- License: Apache-2.0.
- License source: Hugging Face model card metadata.
- Lineage: fine-tuned from SYH99999/MelBandRoformerMergedSYHFTBeta1 (MIT), which builds on KimberleyJSN/melbandroformer (MIT); checked by comparing the weights.
- Students: the teacher and the starting weights of musetric/mel-band-roformer-duality-student (below).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## musetric/mel-band-roformer-duality-student

- Source: https://huggingface.co/musetric/mel-band-roformer-duality-student, revision `c9d2c65a65d820bdd4efdc7d62bb348998969592`.
- Usage: weights (`model.safetensors`) and config (`config.yaml`) of the vocals model the product runs, downloaded as the torch original of the vocals parity reference, and the source of the ONNX core exported by `scripts/onnx/roformer/build_student_onnx.py`.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/parity_audio/vocals_chain.py`, `musetric_toolkit/separate_audio/roformer/pooled_mel_band_roformer.py`, `scripts/onnx/roformer/build_student_onnx.py`, `scripts/onnx/roformer/converting.md`.
- License: Apache-2.0, as the weights of Aname-Tommy/Mel-Band-Roformer_Duality it starts from (above, with the license text).
- Lineage: distilled from Aname-Tommy/Mel-Band-Roformer_Duality by Musetric on openly licensed audio; the shape and the training are described on its model card.

## python-audio-separator

- Source: https://github.com/nomadkaraoke/python-audio-separator
- Usage: Research tool that helped validate the BS-RoFormer approach and integration patterns.
- Local files: `musetric_toolkit/separate_audio/mel_band_roformer_separator.py`, `musetric_toolkit/separate_audio/roformer_utils.py`.
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) 2023 karaokenerds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## WhisperX

- Source: https://github.com/m-bain/whisperX
- Usage: Speech-to-text and word-level alignment for `musetric-transcribe`.
- Local files: `musetric_toolkit/transcribe_audio/whisperx_runner.py`, `musetric_toolkit/transcribe_audio/language_detector.py`, `musetric_toolkit/transcribe_audio/main.py`.
- License: BSD 2-Clause.
- License source: installed package license file and upstream `LICENSE`.

BSD 2-Clause License

Copyright (c) 2024, Max Bain

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## OpenAI Whisper - mobiuslabsgmbh/faster-whisper-large-v3-turbo

- Source: https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
- Usage: base speech-to-text weights behind `musetric-transcribe`. WhisperX loads `large-v3-turbo` through faster-whisper, which downloads this CTranslate2 conversion of `openai/whisper-large-v3-turbo` at runtime.
- Local files: `musetric_toolkit/transcribe_audio/whisperx_runner.py`.
- License: MIT.
- License source: Hugging Face model card metadata.

## OpenAI Whisper - openai/whisper-large-v3-turbo

- Source: https://huggingface.co/openai/whisper-large-v3-turbo, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`.
- Usage: the original weights in torch behind the transcription reference of `musetric-parity`, downloaded at runtime.
- Local files: `musetric_toolkit/parity_audio/transcribe_chain.py`.
- License: MIT.
- License source: the Hugging Face model card metadata at that revision, and the `LICENSE` of https://github.com/openai/whisper. The card declared `apache-2.0` only in its first days; revision `41f01f3f` of 2024-10-04 changed it to `mit`.

MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Transformers.js

- Source: https://github.com/huggingface/transformers.js
- Usage: vendored ONNX conversion scripts (`scripts/convert.py`, `scripts/quantize.py`, `scripts/extra/whisper.py`) used to export Whisper to the transformers.js ONNX layout with cross-attention alignment heads. Only the whisper code path is exercised. The transcription reference of `musetric-parity` ports the token timestamps of its Whisper pipeline (`medianFilter`, `dynamic_time_warping`, `_extract_token_timestamps`).
- Local files: `scripts/onnx/whisper/convert.py`, `scripts/onnx/whisper/quantize.py`, `scripts/onnx/whisper/extra/whisper.py`, `musetric_toolkit/parity_audio/transcribe_chain.py`.
- License: Apache-2.0.
- License source: upstream `LICENSE`.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Local changes, also noted in each file's header: `convert.py` imports `quantize`
as a package-relative module, and the alignment-head table in `extra/whisper.py`
gains a `whisper-large-v3` entry taken from the `openai/whisper-large-v3`
`generation_config.json` (that table carries its own upstream attribution
comment).

## ChordMini

- Source: https://github.com/ptnghia-j/ChordMini
- Usage: vendored inference subset under `musetric_toolkit/chords_audio/chordmini`, plus `2e1d_model_best.pth` checkpoint downloaded at runtime.
- Local files: `musetric_toolkit/chords_audio/chordmini/`, `musetric_toolkit/chords_audio/chordmini_runner.py`, `musetric_toolkit/chords_audio/chordmini_checkpoint.py`.
- License: MIT.
- License source: upstream `LICENSE` and vendored `musetric_toolkit/chords_audio/chordmini/LICENSE`.
- Local vendoring details: see `musetric_toolkit/chords_audio/chordmini/NOTICE.md`.

MIT License

Copyright (c) 2026 ChordMini contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SKey

- Source: https://github.com/deezer/skey
- Usage: musical key detection. The inference subset is vendored; the checkpoint is downloaded at runtime.
- Local files: `musetric_toolkit/key_audio/skey/` (vendored inference code), `musetric_toolkit/key_audio/skey_runner.py`, `musetric_toolkit/key_audio/skey_checkpoint.py`, `musetric_toolkit/key_audio/main.py`.
- License: MIT.
- License source: vendored `musetric_toolkit/key_audio/skey/LICENSE` and upstream `LICENSE`.

MIT License

Copyright (c) 2019-present, Deezer SA.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Beat This!

- Source: https://github.com/CPJKU/beat_this
- Usage: beat and downbeat tracking through the `beat-this` Python package and `beat_this-final0.ckpt` checkpoint.
- Local files: `musetric_toolkit/rhythm_audio/beat_this_runner.py`, `musetric_toolkit/rhythm_audio/main.py`, `musetric_toolkit/rhythm_audio/bpm_estimator.py`.
- License: MIT.
- License source: installed package license file and upstream `LICENSE`.

MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## madmom

- Source: https://github.com/CPJKU/madmom
- Usage: the dynamic Bayesian network downbeat tracker (`DBNDownBeatTrackingProcessor`) behind the Beat This! `dbn` postprocessing, through the `madmom` Python package installed from upstream git.
- Local files: `musetric_toolkit/rhythm_audio/beat_this_runner.py`, `musetric_toolkit/parity_audio/rhythm_chain.py`.
- License: BSD-2-Clause for the source code. The package also installs madmom's data and model files, licensed CC BY-NC-SA 4.0; this project does not use them.
- License source: upstream `LICENSE`.

Copyright (c) 2012-2014 Department of Computational Perception,
Johannes Kepler University, Linz, Austria and Austrian Research Institute for
Artificial Intelligence (OFAI), Vienna, Austria.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## RMVPE

- Source: https://github.com/Dream-High/RMVPE (paper arXiv:2306.15412) and https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI (network definition and inference).
- Usage: reference F0 extraction for measuring the Musetric pitch tracker (`musetric-pitch`); not part of any processing step of the app. The network definition is vendored; the checkpoint `rmvpe.pt` is downloaded at runtime from https://huggingface.co/lj1995/VoiceConversionWebUI, where it was trained and published by the RVC project (yxlllc and RVC-Boss) on MIR-1K, PTDB-TUG and M4Singer synthesis data.
- Local files: `musetric_toolkit/pitch_audio/rmvpe/` (vendored network), `musetric_toolkit/pitch_audio/tracker.py`, `musetric_toolkit/common/envs.py`.
- License: MIT for the RVC code and the Hugging Face repository; Apache-2.0 for the original RMVPE repository.
- License source: vendored `musetric_toolkit/pitch_audio/rmvpe/LICENSE`, upstream `LICENSE` and the Hugging Face model card metadata.

MIT License

Copyright (c) 2023 liujing04
Copyright (c) 2023 源文雨
Copyright (c) 2023 Ftps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## CREPE (torchcrepe)

- Source: https://github.com/maxrmorrison/torchcrepe, a PyTorch port of https://github.com/marl/crepe.
- Usage: one of the four models that `musetric-pitch` combines into the reference of the Musetric pitch bench, and a model of `musetric-pitch-zoo`, which checks that reference; not part of any processing step of the app. The `full` weights ship inside the `torchcrepe` package; its salience is decoded with the global Viterbi of `musetric_toolkit/pitch_audio/tracker.py`, because the package's own decoders add random dither of up to one bin (20 cents) and decode each batch apart.
- Local files: `musetric_toolkit/pitch_zoo/crepe_model.py`, `musetric_toolkit/pitch_audio/main.py`.
- License: MIT.
- License source: installed package license file.

MIT License

Copyright (c) 2020 Max Morrison

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SwiftF0

- Source: https://github.com/lars76/swift-f0
- Usage: one of the four models that `musetric-pitch` combines into the reference of the Musetric pitch bench, and a model of `musetric-pitch-zoo`, which checks that reference; not part of any processing step of the app. The ONNX model ships inside the `swift-f0` package.
- Local files: `musetric_toolkit/pitch_zoo/swiftf0_model.py`, `musetric_toolkit/pitch_audio/main.py`.
- License: MIT.
- License source: installed package license file.

MIT License

Copyright (c) 2025-2026 Lars Nieradzik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FCPE (torchfcpe)

- Source: https://github.com/CNChTu/FCPE
- Usage: one of the four models that `musetric-pitch` combines into the reference of the Musetric pitch bench, and a model of `musetric-pitch-zoo`, which checks that reference; not part of any processing step of the app. The weights (`fcpe_c_v001.pt`) ship inside the `torchfcpe` package.
- Local files: `musetric_toolkit/pitch_zoo/fcpe_model.py`, `musetric_toolkit/pitch_audio/main.py`.
- License: MIT.
- License source: installed package license file.

MIT License

Copyright (c) 2023 CN_ChiTu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PENN (FCNF0++)

- Source: https://github.com/interactiveaudiolab/penn; checkpoint https://huggingface.co/maxrmorrison/fcnf0-plus-plus, revision `74911e26f43ad38790a42592e77f9d8be0a5dd1c`.
- Usage: a model of `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app. The checkpoint `fcnf0++.pt` is downloaded at runtime.
- Local files: `musetric_toolkit/pitch_zoo/penn_model.py`.
- License: MIT.
- License source: installed package license file and the Hugging Face model card metadata.

MIT License

Copyright (c) 2022 Interactive Audio Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PESTO

- Source: https://github.com/SonyCSLParis/pesto
- Usage: a model of `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app. The `mir-1k_g7` checkpoint ships inside the `pesto-pitch` package, which is used unmodified as a library.
- Local files: `musetric_toolkit/pitch_zoo/pesto_model.py`.
- License: LGPL-3.0.
- License source: installed package license file (`LICENSE.md`).

## WORLD (pyworld)

- Source: https://github.com/mmorise/World (the vocoder), https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder (the `pyworld` wrapper), built for current Python by https://github.com/tsukumijima/pyworld-prebuilt.
- Usage: `musetric-pitch-zoo resynth` resynthesizes vocals along a known f0 curve (CheapTrick, D4C and the WORLD synthesizer) to make a ground-truth set, used by `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app.
- Local files: `musetric_toolkit/pitch_zoo/resynth.py`.
- License: modified BSD (WORLD, Copyright (c) 2010 M. Morise); MIT (pyworld, Copyright 2016 pyworld contributors).
- License source: upstream `LICENSE.txt` of WORLD and the installed package license file.

## vocadito

- Source: https://zenodo.org/records/5578807 (R. Bittner, K. Pasalo, J. J. Bosch, G. Meseguer Brocal, D. Rubinstein: vocadito: A dataset of solo vocals with f0, note, and lyric annotations, 2021).
- Usage: 40 solo singing excerpts and their f0 annotations, downloaded at runtime by `musetric-pitch-zoo truth` as ground truth for `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app.
- Local files: `musetric_toolkit/pitch_zoo/truth.py`.
- License: CC BY 4.0.
- License source: Zenodo record metadata.

## Dagstuhl ChoirSet

- Source: https://zenodo.org/records/4618287, version 1.2.3 (S. Rosenzweig, H. Cuesta, C. Weiß, F. Scherbaum, E. Gómez, M. Müller: Dagstuhl ChoirSet: A Multitrack Dataset for MIR Research on Choral Singing, TISMIR 3(1), 2020).
- Usage: the eight voices with manually annotated f0 and their dynamic-microphone recordings, read at runtime from the archive by `musetric-pitch-zoo truth` as ground truth for `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app.
- Local files: `musetric_toolkit/pitch_zoo/truth.py`.
- License: CC BY 4.0.
- License source: Zenodo record metadata.

## PTDB-TUG

- Source: https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html (G. Pirker, M. Wohlmayr, S. Petrik, F. Pernkopf: A Pitch Tracking Corpus with Evaluation on Multipitch Tracking Scenario, Interspeech 2011).
- Usage: microphone recordings and laryngograph-based reference pitch of 100 utterances, downloaded at runtime by `musetric-pitch-zoo truth` as ground truth for `musetric-pitch-zoo`, which checks the pitch reference of the Musetric pitch bench; not part of any processing step of the app.
- Local files: `musetric_toolkit/pitch_zoo/truth.py`.
- License: Open Database License 1.0 for the database, Database Contents License 1.0 for its contents.
- License source: the database page.
