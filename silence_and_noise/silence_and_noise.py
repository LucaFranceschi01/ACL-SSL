import os
import torch, torchaudio
from util import get_prompt_template

def process_audio(audio_file, SAMPLE_RATE = 16000, set_length: int = 10):
    if audio_file.shape[0] > 1:
        audio_file = audio_file.mean(dim=0)

    audio_file = audio_file.squeeze(0)

    # slicing or padding based on set_length
    # slicing
    if audio_file.shape[0] > (SAMPLE_RATE * set_length):
        audio_file = audio_file[:SAMPLE_RATE * set_length]
    # zero padding
    if audio_file.shape[0] < (SAMPLE_RATE * set_length):
        pad_len = (SAMPLE_RATE * set_length) - audio_file.shape[0]
        pad_val = torch.zeros(pad_len)
        audio_file = torch.cat((audio_file, pad_val), dim=0)

    return audio_file

def get_real_noise_audios(real_san_audios_path) -> torch.Tensor:
    audio_paths = os.listdir(real_san_audios_path)
    audio_files = []

    for audio_path in audio_paths:
        audio_file, _ = torchaudio.load(os.path.join(real_san_audios_path, audio_path))
        audio_files.append(process_audio(audio_file))

    return torch.stack(audio_files, dim=0)

def get_silence_noise_audios(module, audio_size, real_san_audios_path = None):
    if real_san_audios_path != None:
        negative_audios = torch.cat(
            (
                torch.zeros(audio_size), # silence
                torch.clip(torch.randn(audio_size), min=-1., max=1.), # gaussian noise
                get_real_noise_audios(real_san_audios_path)
            ),
            dim=0
        )
    else:
        negative_audios = torch.cat(
            (
                torch.zeros(audio_size), # silence
                torch.clip(torch.randn(audio_size), min=-1., max=1.), # gaussian noise
            ),
            dim=0
        )

    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()
    placeholder_tokens = module.get_placeholder_token(prompt_template.replace('{}', ''))
    placeholder_tokens = placeholder_tokens.repeat((negative_audios.shape[0], 1))

    with torch.no_grad():
        neg_audios_embedded = module.encode_audio(negative_audios.to(module.device),
                                                  placeholder_tokens, text_pos_at_prompt, prompt_length)

    return neg_audios_embedded.detach() # torch.Size([negative_audios.shape[0], 512])