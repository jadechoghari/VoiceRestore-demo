import torch
import torchaudio
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import torch
from BigVGAN import bigvgan
from BigVGAN.meldataset import get_mel_spectrogram
from voice_restore import VoiceRestore
import argparse
from model import OptimizedAudioRestorationModel
import librosa
from inference_long import apply_overlap_windowing_waveform, reconstruct_waveform_from_windows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration class for VoiceRestore
class VoiceRestoreConfig(PretrainedConfig):
    model_type = "voice_restore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get("steps", 16)
        self.cfg_strength = kwargs.get("cfg_strength", 0.5)
        self.window_size_sec = kwargs.get("window_size_sec", 5.0)
        self.overlap = kwargs.get("overlap", 0.5)

# Model class for VoiceRestore
class VoiceRestore(PreTrainedModel):
    config_class = VoiceRestoreConfig
    
    def __init__(self, config: VoiceRestoreConfig):
        super().__init__(config)
        self.steps = config.steps
        self.cfg_strength = config.cfg_strength
        self.window_size_sec = config.window_size_sec
        self.overlap = config.overlap

        # Initialize BigVGAN model
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_v2_24khz_100band_256x',
            use_cuda_kernel=False,
            force_download=False
        ).to(device)
        self.bigvgan_model.remove_weight_norm()

        # Optimized restoration model
        self.optimized_model = OptimizedAudioRestorationModel(device=device, bigvgan_model=self.bigvgan_model)
        save_path = "./pytorch_model.bin"
        state_dict = torch.load(save_path, map_location=torch.device(device))
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        self.optimized_model.voice_restore.load_state_dict(state_dict, strict=True)
        self.optimized_model.eval()

    def forward(self, input_path, output_path, short=True):
        # Restore the audio using the parameters from the config
        if short:
            self.restore_audio_short(self.optimized_model, input_path, output_path, self.steps, self.cfg_strength)
        else:
            self.restore_audio_long(self.optimized_model, input_path, output_path, self.steps, self.cfg_strength, self.window_size_sec, self.overlap)

    def restore_audio_short(self, model, input_path, output_path, steps, cfg_strength):
        """
        Short inference for audio restoration.
        """
        # Load the audio file
        device_type = device.type  
        audio, sr = torchaudio.load(input_path)
        if sr != model.target_sample_rate:
            audio = torchaudio.functional.resample(audio, sr, model.target_sample_rate)

        audio = audio.mean(dim=0, keepdim=True) if audio.dim() > 1 else audio  # Convert to mono if stereo

        with torch.inference_mode():
            with torch.autocast(device_type):
                restored_wav = model(audio, steps=steps, cfg_strength=cfg_strength)
                restored_wav = restored_wav.squeeze(0).float().cpu()  # Move to CPU after processing

        # Save the restored audio
        torchaudio.save(output_path, restored_wav, model.target_sample_rate)

    def restore_audio_long(self, model, input_path, output_path, steps, cfg_strength, window_size_sec, overlap):
        """
        Long inference for audio restoration using overlapping windows.
        """
        # Load the audio file
        wav, sr = librosa.load(input_path, sr=24000, mono=True)
        wav = torch.FloatTensor(wav).unsqueeze(0)  # Shape: [1, num_samples]

        window_size_samples = int(window_size_sec * sr)
        wav_windows = apply_overlap_windowing_waveform(wav, window_size_samples, overlap)

        restored_wav_windows = []
        for wav_window in wav_windows:
            wav_window = wav_window.to(device)
            processed_mel = get_mel_spectrogram(wav_window, self.bigvgan_model.h).to(device)

            # Restore audio
            with torch.no_grad():
                with torch.autocast(device):
                    restored_mel = model.voice_restore.sample(processed_mel.transpose(1, 2), steps=steps, cfg_strength=cfg_strength)
                    restored_mel = restored_mel.squeeze(0).transpose(0, 1)

                restored_wav = self.bigvgan_model(restored_mel.unsqueeze(0)).squeeze(0).float().cpu()
                restored_wav_windows.append(restored_wav)

            torch.cuda.empty_cache()

        restored_wav_windows = torch.stack(restored_wav_windows)
        restored_wav = reconstruct_waveform_from_windows(restored_wav_windows, window_size_samples, overlap)

        # Save the restored audio
        torchaudio.save(output_path, restored_wav.unsqueeze(0), 24000)


# # Function to load the model using AutoModel
# from transformers import AutoModel

# def load_voice_restore_model(checkpoint_path: str):
#     model = AutoModel.from_pretrained(checkpoint_path, config=VoiceRestoreConfig())
#     return model

# # Example Usage
# model = load_voice_restore_model("./checkpoints/voice-restore-20d-16h-optim.pt")
# model("test_input.wav", "test_output.wav")
