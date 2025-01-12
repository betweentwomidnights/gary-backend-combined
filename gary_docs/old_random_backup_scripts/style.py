import torch
import torchaudio
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import argparse

def setup_model(eval_q: int = 1, excerpt_length: float = 3.0, duration: float = 10.0,
                ds_factor: int = None, encodec_n_q: int = None):
    """Initialize and configure the MusicGen-Style model."""
    print("Loading MusicGen-Style model...")
    model = MusicGen.get_pretrained('facebook/musicgen-style')
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        use_sampling=True,
        top_k=250,
        cfg_coef=3.0,
        cfg_coef_beta=5.0
    )
    
    # Set style conditioner parameters with new options
    model.set_style_conditioner_params(
        eval_q=eval_q,
        excerpt_length=excerpt_length,
        ds_factor=ds_factor,        # New: controls token downsampling
        encodec_n_q=encodec_n_q     # New: controls feature extraction
    )
    
    return model

def process_audio(input_path: str, output_dir: str, num_variations: int = 3, 
                 eval_q: int = 1, excerpt_length: float = 3.0, duration: float = 10.0,
                 description: str = None, ds_factor: int = None, encodec_n_q: int = None):
    """Process input audio and generate style-transferred variations."""
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Input file: {input_path}")
    print(f"- Excerpt length: {excerpt_length} seconds")
    print(f"- Output duration: {duration} seconds")
    print(f"- Style adherence (eval_q): {eval_q}")
    print(f"- Downsampling factor: {ds_factor if ds_factor else 'default'}")
    print(f"- EnCodec streams: {encodec_n_q if encodec_n_q else 'default'}")
    print(f"- Description: {description if description else 'None'}\n")
    
    # Load input audio
    print("Loading input audio...")
    audio_source, sr = torchaudio.load(input_path)
    
    # Initialize model
    model = setup_model(eval_q, excerpt_length, duration, ds_factor, encodec_n_q)
    
    # Generate variations one at a time
    for idx in range(num_variations):
        print(f"\nGenerating variation {idx + 1}/{num_variations}")
        
        output = model.generate_with_chroma(
            descriptions=[description],
            melody_wavs=[audio_source],
            melody_sample_rate=sr,
            progress=True
        )
        
        # Save the output
        output_path = output_dir / f"style_variation_{idx}.wav"
        print(f"Saving to {output_path}")
        audio_write(
            str(output_path),
            output[0].cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )
        
        # Clear CUDA cache between generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Generate style variations of an audio file using MusicGen-Style")
    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument("--output_dir", type=str, default="./style_outputs", help="Output directory")
    parser.add_argument("--variations", type=int, default=3, help="Number of variations to generate")
    parser.add_argument("--eval_q", type=int, default=1, choices=range(1, 7),
                      help="Style adherence (1-6, lower = less strict)")
    parser.add_argument("--excerpt_length", type=float, default=10.0, 
                      help="Length of input audio to analyze (1.5-4.5 seconds)")
    parser.add_argument("--duration", type=float, default=30.0,
                      help="Length of output audio in seconds (up to 30)")
    parser.add_argument("--description", type=str, default=None,
                      help="Text description to guide the style transfer")
    parser.add_argument("--ds_factor", type=int, default=None,
                      help="Downsampling factor for style tokens")
    parser.add_argument("--encodec_n_q", type=int, default=None,
                      help="Number of EnCodec streams for feature extraction")
    
    args = parser.parse_args()
    
    process_audio(
        args.input,
        args.output_dir,
        args.variations,
        args.eval_q,
        args.excerpt_length,
        args.duration,
        args.description,
        args.ds_factor,
        args.encodec_n_q
    )

if __name__ == "__main__":
    main()