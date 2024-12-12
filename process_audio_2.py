import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from tqdm import tqdm
import csv
import numpy as np

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

class AudioAugmenter:
    def __init__(self, num_tries=5, batch_size=4):
        self.num_tries = num_tries
        self.batch_size = batch_size

    def pregenerate_augmentation_params(self, num_files):
        total_augs = num_files * self.num_tries

        params = {
            'pitch_shift': np.random.uniform(-2, 2, total_augs),
            'noise_scale': np.random.uniform(0.01, 0.05, total_augs),
            'gain_db': np.random.uniform(-3, 3, total_augs)
        }

        params['gain_factor'] = 10 ** (params['gain_db'] / 20.0)

        # Randomly choose which effect to use for each augmentation
        effects = ['pitch_shift', 'add_noise', 'gain']
        params['chosen_effects'] = np.random.choice(effects, total_augs)

        return params

    @torch.no_grad()
    def augment_batch(self, audio_batch, sr_batch, aug_params, batch_idx):
        """Apply augmentations to a batch of audio files"""
        idx = batch_idx * len(audio_batch)
        effect = aug_params['chosen_effects'][idx]

        if effect == 'pitch_shift':
            shift = aug_params['pitch_shift'][idx]
            if USE_GPU:
                return torchaudio.functional.pitch_shift(audio_batch.cpu(), sr_batch, shift).to(device)
            return torchaudio.functional.pitch_shift(audio_batch, sr_batch, shift)

        elif effect == 'add_noise':
            scale = aug_params['noise_scale'][idx]
            noise = torch.randn_like(audio_batch)
            return audio_batch + noise * scale

        else:
            gain = aug_params['gain_factor'][idx]
            return audio_batch * gain

def process_data(data_folder, output_folder, tries=5, batch_size=4):
    input_folder = os.path.join(data_folder, "output1")
    output_folder = os.path.join(output_folder, "output2")
    os.makedirs(output_folder, exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    hum_out = os.path.join(output_folder, "hum")
    song_out = os.path.join(output_folder, "song")
    os.makedirs(hum_out, exist_ok=True)
    os.makedirs(song_out, exist_ok=True)

    augmenter = AudioAugmenter(tries, batch_size)

    with open(metadata_path, newline='') as csvfile:
        total_files = sum(1 for _ in csv.DictReader(csvfile))

    # Pre-generate all augmentation parameters
    aug_params = augmenter.pregenerate_augmentation_params(total_files)

    output_metadata = []
    current_batch = {'hum': [], 'song': [], 'metadata': []}

    with torch.no_grad():
        with open(metadata_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row_idx, row in enumerate(tqdm(reader, total=total_files, desc="Processing files")):
                hum_path = os.path.join(input_folder, "hum", row['hum'])
                song_path = os.path.join(input_folder, "song", row['song'])

                hum_audio, hum_sr = torchaudio.load(hum_path)
                song_audio, song_sr = torchaudio.load(song_path)

                if USE_GPU:
                    hum_audio = hum_audio.to(device)
                    song_audio = song_audio.to(device)

                current_batch['hum'].append(hum_audio)
                current_batch['song'].append(song_audio)
                current_batch['metadata'].append({
                    'id': row['id'],
                    'hum_base': os.path.splitext(row['hum'])[0],
                    'song_base': os.path.splitext(row['song'])[0],
                    'info': row['info'],
                    'sr': (hum_sr, song_sr)
                })

                if len(current_batch['hum']) == batch_size or row_idx == total_files - 1:
                    # Process original files
                    for idx, meta in enumerate(current_batch['metadata']):
                        hum_original = f"{meta['hum_base']}.mp3"
                        song_original = f"{meta['song_base']}.mp3"

                        if not os.path.exists(os.path.join(hum_out, hum_original)):
                            torchaudio.save(
                                os.path.join(hum_out, hum_original),
                                current_batch['hum'][idx].cpu(),
                                meta['sr'][0],
                                format='mp3'
                            )

                        if not os.path.exists(os.path.join(song_out, song_original)):
                            torchaudio.save(
                                os.path.join(song_out, song_original),
                                current_batch['song'][idx].cpu(),
                                meta['sr'][1],
                                format='mp3'
                            )

                        output_metadata.append((meta['id'], hum_original, song_original, meta['info']))

                    # Process augmentations
                    for aug_idx in range(tries):
                        for idx, meta in enumerate(current_batch['metadata']):
                            hum_aug = f"{meta['hum_base']}_aug{aug_idx}.mp3"
                            song_aug = f"{meta['song_base']}_aug{aug_idx}.mp3"

                            if os.path.exists(os.path.join(hum_out, hum_aug)) and \
                               os.path.exists(os.path.join(song_out, song_aug)):
                                output_metadata.append((meta['id'], hum_aug, song_aug, meta['info']))
                                continue

                            aug_hum = augmenter.augment_batch(
                                current_batch['hum'][idx],
                                meta['sr'][0],
                                aug_params,
                                row_idx * tries + aug_idx
                            )

                            aug_song = augmenter.augment_batch(
                                current_batch['song'][idx],
                                meta['sr'][1],
                                aug_params,
                                row_idx * tries + aug_idx
                            )

                            torchaudio.save(
                                os.path.join(hum_out, hum_aug),
                                aug_hum.cpu(),
                                meta['sr'][0],
                                format='mp3'
                            )

                            torchaudio.save(
                                os.path.join(song_out, song_aug),
                                aug_song.cpu(),
                                meta['sr'][1],
                                format='mp3'
                            )

                            output_metadata.append((meta['id'], hum_aug, song_aug, meta['info']))

                    if USE_GPU:
                        torch.cuda.empty_cache()
                    current_batch = {'hum': [], 'song': [], 'metadata': []}

                    if len(output_metadata) % (batch_size * 10) == 0:
                        save_metadata(output_metadata, output_folder)

    save_metadata(output_metadata, output_folder)
    print("Processing complete.")

def save_metadata(output_metadata, output_folder):
    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song", "info"])
        writer.writerows(output_metadata)
