"""
Configuration file containing all available variations for MelodyFlow transformations.
Each variation includes a descriptive prompt and processing parameters.
"""

VARIATIONS = {
    # Acoustic Instruments
    'accordion_folk': {
        'prompt': "Lively accordion music with a European folk feeling, perfect for a travel documentary about traditional culture and street performances in Paris",
        'flowstep': 0.13,
        'steps': 10
    },
    'banjo_bluegrass': {
        'prompt': "Authentic bluegrass banjo band performance with rich picking patterns, ideal for a heartfelt documentary about American rural life and traditional crafts",
        'flowstep': 0.13,
        'steps': 25
    },
    'piano_classical': {
        'prompt': "Expressive classical piano performance with dynamic range and emotional depth, ideal for a luxury brand commercial",
        'flowstep': 0.13,
        'steps': 25
    },
    'celtic': {
        'prompt': "Traditional Celtic arrangement with fiddle and flute, perfect for a documentary about Ireland's stunning landscapes and ancient traditions",
        'flowstep': 0.13,
        'steps': 25
    },
    'strings_quartet': {
        'prompt': "Elegant string quartet arrangement with rich harmonies and expressive dynamics, perfect for wedding ceremony music",
        'flowstep': 0.13,
        'steps': 25
    },

    # Synthesizer Variations
    'synth_retro': {
        'prompt': "1980s style synthesizer melody with warm analog pads and arpeggios, perfect for a nostalgic sci-fi movie soundtrack",
        'flowstep': 0.13,
        'steps': 25
    },
    'synth_modern': {
        'prompt': "Modern electronic production with crisp digital synthesizers and vocoder effects, ideal for a tech product launch video",
        'flowstep': 0.13,
        'steps': 25
    },
    'synth_ambient': {
        'prompt': "Atmospheric synthesizer pads with reverb and delay, perfect for a meditation app or wellness commercial",
        'flowstep': 0.13,
        'steps': 25
    },
    'synth_edm': {
        'prompt': "High-energy EDM synth leads with sidechain compression and modern production, perfect for sports highlights or action sequences",
        'flowstep': 0.13,
        'steps': 25
    },

    # Band Arrangements
    'rock_band': {
        'prompt': "Full rock band arrangement with electric guitars, bass, and drums, perfect for an action movie trailer",
        'flowstep': 0.13,
        'steps': 25
    },

    # Hybrid/Special
    'cinematic_epic': {
        'prompt': "Epic orchestral arrangement with modern hybrid elements, synthesizers, and percussion, perfect for movie trailers",
        'flowstep': 0.12,
        'steps': 25
    },
    'lofi_chill': {
        'prompt': "Lo-fi hip hop style with vinyl crackle, mellow piano, and tape saturation, perfect for study or focus playlists",
        'flowstep': 0.12,
        'steps': 25
    },
    'synth_bass': {
        'prompt': "Deep analog synthesizer bassline with modern production and subtle modulation, perfect for electronic music production",
        'flowstep': 0.12,
        'steps': 25
    },

    'retro_rpg': {
        'prompt': "16-bit era JRPG soundtrack with bright melodic synthesizers, orchestral elements, and adventurous themes, perfect for a fantasy video game battle scene or overworld exploration",
        'flowstep': 0.13,
        'steps': 25
    },

    'steel_drums': {
        'prompt': "Vibrant Caribbean steel drum ensemble with tropical percussion and uplifting melodies, perfect for a beach resort commercial or travel documentary",
        'flowstep': 0.13,
        'steps': 25
    },
    'chiptune': {
        'prompt': "8-bit video game soundtrack with arpeggiated melodies and classic NES-style square waves, perfect for a retro platformer or action game",
        'flowstep': 0.13,
        'steps': 25
    },
    'gamelan_fusion': {
        'prompt': "Indonesian gamelan ensemble with metallic percussion, gongs, and ethereal textures, perfect for a meditation app or spiritual documentary",
        'flowstep': 0.13,
        'steps': 25
    },

    'music_box': {
        'prompt': "Delicate music box melody with gentle bell tones and ethereal ambiance, perfect for a children's lullaby or magical fantasy scene",
        'flowstep': 0.12,
        'steps': 25
    }
}