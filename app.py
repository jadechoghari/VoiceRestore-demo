import torch
import gradio as gr
import torchaudio
from transformers import AutoModel
import spaces


checkpoint_path = "./"
model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)


@spaces.GPU()
def restore_audio(input_audio):
    # load the audio file
    output_path = "restored_output.wav"
    model(input_audio, output_path)
    return output_path


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🔊 Voice Restoration with Transformer-based Model</h1>")
    gr.Markdown(
        """
        <p style='text-align: center;'>Upload a degraded audio file or select an example, and the space will restore it using the <b>VoiceRestore</b> model!<br>
        Based on this <a href='https://github.com/skirdey/voicerestore' target='_blank'>repo</a> by <a href='https://github.com/skirdey' target='_blank'>@Stan Kirdey</a>,<br>
        and the HF Transformers 🤗 model by <a href='https://github.com/jadechoghari' target='_blank'>@jadechoghari</a>.
        </p>
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎧 Select an Example or Upload Your Audio:")
            input_audio = gr.Audio(label="Upload Degraded Audio", type="filepath")
            gr.Examples(
                examples=["example_input.wav", "example_16khz.wav", "example-distort-16khz.wav", "example-full-degrad.wav", "example-reverb-16khz.wav"],
                inputs=input_audio,
                label="Sample Degraded Audios"
            ),
            cache_examples="lazy"
        
        with gr.Column():
            gr.Markdown("### 🎶 Restored Audio Output:")
            output_audio = gr.Audio(label="Restored Audio", type="filepath")

    with gr.Row():
        restore_btn = gr.Button("✨ Restore Audio")

    # Connect the button to the function
    restore_btn.click(restore_audio, inputs=input_audio, outputs=output_audio)

# Launch the demo
demo.launch(debug=True)
