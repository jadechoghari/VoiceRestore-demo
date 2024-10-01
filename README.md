---
title: VoiceRestore
emoji: 🔊
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.0.0b3
app_file: app.py
pinned: false
---
# VoiceRestore: Flow-Matching Transformers for Speech Recording Quality Restoration

VoiceRestore is a cutting-edge speech restoration model designed to significantly enhance the quality of degraded voice recordings. Leveraging flow-matching transformers, this model excels at addressing a wide range of audio imperfections commonly found in speech, including background noise, reverberation, distortion, and signal loss.

It is based on this [repo](https://github.com/skirdey/voicerestore) & demo of audio restorations: [VoiceRestore](https://sparkling-rabanadas-3082be.netlify.app/)

## Build - using Gradio 🟠
``` bash
!git lfs install
!git clone https://github.com/jadechoghari/VoiceRestore-demo
%cd VoiceRestore
!pip install -r requirements.txt
!python app.py
```

## Usage - using Transformers 🤗
``` bash
!git lfs install
!git clone https://huggingface.co/jadechoghari/VoiceRestore
%cd VoiceRestore
!pip install -r requirements.txt
```

``` python
from transformers import AutoModel
# path to the model folder (on colab it's as follows)
checkpoint_path = "/content/VoiceRestore"
model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
model("test_input.wav", "test_output.wav")
```




## Example
### Degraded Input: 

### Degraded Input Audio

<audio controls>
  <source src="https://huggingface.co/jadechoghari/VoiceRestore/resolve/main/test_input.wav" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

---
### Restored (steps=32, cfg=1.0):

<audio controls>
  <source src="https://huggingface.co/jadechoghari/VoiceRestore/resolve/main/test_output.wav" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

Restored audio - 16 steps, strength 0.5:

---
## Key Features

- **Universal Restoration**: The model can handle any level and type of voice recording degradation. Pure magic.  
- **Easy to Use**: Simple interface for processing degraded audio files.
- **Pretrained Model**: Includes a 301 million parameter transformer model with pre-trained weights. (Model is still in the process of training, there will be further checkpoint updates)

---


## Model Details

- **Architecture**: Flow-matching transformer
- **Parameters**: 300M+ parameters
- **Input**: Degraded speech audio (various formats supported)
- **Output**: Restored speech

## Limitations and Future Work

- Current model is optimized for speech; may not perform optimally on music or other audio types.
- Ongoing research to improve performance on extreme degradations.
- Future updates may include real-time processing capabilities.

## Citation

If you use VoiceRestore in your research, please cite our paper:

```
@article{kirdey2024voicerestore,
  title={VoiceRestore: Flow-Matching Transformers for Speech Recording Quality Restoration},
  author={Kirdey, Stanislav},
  journal={arXiv},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the [E2-TTS implementation by Lucidrains](https://github.com/lucidrains/e2-tts-pytorch)
- Special thanks to the open-source community for their invaluable contributions.
- Credits: This repository is based on the [E2-TTS implementation by Lucidrains](https://github.com/lucidrains/e2-tts-pytorch)
