# Third-Party Audio Provenance

## f1_start_light_beep.wav

- Source: `F1 start`, a community upload by TyLenol32 on Voicemod Tuna
- Page: https://tuna.voicemod.net/sound/6371c440-a5e2-43f7-af25-8142a778d726
- Public audio: https://content-storage.voicemod.net/d724c68ee9a492762ef2d3756f2360f60961b1fa1430b805857a6676501cb77f
- Retrieved: 2026-09-28
- The uploader describes it as F1 starting lights. Official Formula 1
  provenance and a separate redistribution license are not established by this page.

The downloaded MP3 is decoded to 16-bit PCM WAV with macOS `afconvert`.
`scripts/process_start_light_beep.py` extracts 0.735–1.105 seconds (one beep),
mixes to mono, removes DC offset, normalizes to 85% peak, and applies short
edge fades. The packaged 48 kHz WAV plays once per red light; the source
recording is not packaged. To reproduce after downloading the MP3:

```sh
afconvert -f WAVE -d LEI16 f1-start-source.mp3 f1-start-source.wav
python scripts/process_start_light_beep.py f1-start-source.wav src/racing/assets/audio/f1_start_light_beep.wav
```

## formula_engine_body_loop.wav

This formula engine layer is an original generated audio asset created by
`scripts/generate_formula_engine_assets.py`. It is a deterministic mono
22.05 kHz PCM WAV file synthesized from harmonic, pulse, and seeded-noise
layers; no third-party samples are used.

## f1_engine_loop.wav

- Source: Pixabay sound effect 32824, `f1`
- URL: https://pixabay.com/sound-effects/musical-f1-32824/
- Author: freesound_community
- License: Pixabay Content License
- License summary: https://pixabay.com/service/license-summary/
- Source file used: `freesound_community-f1-32824.mp3`

The bundled loop is derived from a user-downloaded Pixabay MP3 source. The
source MP3 is decoded into a temporary mono WAV, then the processor selects a
stable high-energy 3.2 second engine window, smooths sharp spikes, applies a
gentle low-pass blend, stabilizes short amplitude-envelope lumps, converts it
to mono 22.05 kHz PCM WAV, removes DC offset, crossfades the tail into the
head, matches the final loop seam, and normalizes conservatively below
clipping. The MP3 source is intentionally not packaged with the game.

## tire_squeal_1.wav, tire_squeal_2.wav, tire_squeal_3.wav

- Source: Pixabay sound effect 14764, `city chrysler lhs tire squeal 02 04 25 2009wav`
- URL: https://pixabay.com/sound-effects/city-chrysler-lhs-tire-squeal-02-04-25-2009wav-14764/
- Author: freesound_community
- License: Pixabay Content License
- License summary: https://pixabay.com/service/license-summary/
- Source file used: `freesound_community-chrysler-lhs-tire-squeal-03-04-25-2009-7154.mp3`

The bundled clips are derived from a user-downloaded Pixabay MP3 source. The
source MP3 is decoded into a temporary mono WAV, then the processor selects
three distinct high-frequency squeal regions, converts each to mono 22.05 kHz
PCM WAV, removes DC offset, crossfades each tail into its head, and normalizes
below clipping. Each clip also gets a short loop-seam match so runtime volume
tapers can enter and exit without edge clicks. The MP3 source is intentionally
not packaged with the game.

## berlin_town_music.wav

- Source: Freesound sound 564665, `Berlin Town`
- URL: https://freesound.org/people/kjartan_abel/sounds/564665/
- Author: kjartan_abel / Kjartan Abel
- License: Creative Commons Attribution 4.0 on the Freesound license field; the sound description requests CC BY-SA 4.0 attribution language
- Attribution text: Berlin Town by Kjartan Abel. Visit https://kjartan-abel.com/library to download royalty-free music for your next project.
- Preview used: https://cdn.freesound.org/previews/564/564665_6738752-hq.mp3

The original WAV download requires a Freesound login, so the bundled loop is
derived from the public Freesound preview. The processor converts it to mono
22.05 kHz PCM WAV, removes DC offset, trims the quiet lead-in to the musical
onset, fades in over the first five seconds, crossfades the tail into the
head, and normalizes below clipping.
