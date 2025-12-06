#!/usr/bin/env python3
"""
Process the FreeSound-LAION-640k dataset:
1. Download from HuggingFace
2. Filter clips (<20s, no speech/singing/music)
3. Extract embeddings with ONNX model (GPU)
4. Upload to Pinecone
"""

import os
import json
import getpass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm
from pinecone import Pinecone

# --- Configuration ---
MODEL_PATH = "model_v1.onnx"
OUTPUT_JSON = "freesound_embeddings.json"
INDEX_NAME = "imitune-search"
SAMPLE_RATE = 16000  # Standard for audio embeddings
MAX_DURATION_SECONDS = 20
CLIP_DURATION_SECONDS = 10  # Extract embeddings from first 10 seconds
BATCH_SIZE = 32  # For GPU inference

# AudioSet ontology tags to filter out (speech, singing, music related)
# These are based on the AudioSet ontology - we match against lowercase tag names
EXCLUDED_TAGS = {
    # Human voice/speech related
    "speech", "speaking", "talk", "talking", "voice", "voices", "vocal", "vocals",
    "male speech", "female speech", "child speech", "man speaking", "woman speaking",
    "kid speaking", "conversation", "narration", "monologue", "babbling",
    "speech synthesizer", "shout", "shouting", "scream", "screaming", "yell", "yelling",
    "whisper", "whispering", "laughter", "laugh", "laughing", "giggle", "chuckle",
    "cry", "crying", "sobbing", "whimper", "sigh", "humming",
    "crowd", "chatter", "hubbub", "speech noise", "speech babble",
    "children shouting", "cheering",
    
    # Singing related
    "singing", "sing", "singer", "song", "vocal music", "a capella", "acapella",
    "choir", "choral", "yodeling", "chant", "chanting", "mantra",
    "male singing", "female singing", "child singing", "synthetic singing",
    "rapping", "rap", "rapper", "beatbox", "beatboxing",
    "opera", "lullaby",
    
    # Music related
    "music", "musical", "melody", "melodic", "harmonic", "harmony",
    "instrument", "instrumental", "orchestra", "orchestral", "band",
    "guitar", "acoustic guitar", "electric guitar", "bass guitar",
    "piano", "keyboard", "organ", "synthesizer", "synth",
    "violin", "fiddle", "viola", "cello", "double bass", "string",
    "trumpet", "trombone", "horn", "french horn", "tuba", "brass",
    "flute", "clarinet", "oboe", "bassoon", "saxophone", "sax", "woodwind",
    "drum", "drums", "drummer", "drumming", "percussion", "percussive",
    "cymbal", "hi-hat", "snare", "kick", "tom",
    "bass", "beat", "beats", "rhythm", "rhythmic",
    "chord", "chords", "riff", "solo",
    "rock", "pop", "jazz", "blues", "classical", "electronic", "techno",
    "house", "hip-hop", "hip hop", "r&b", "country", "folk", "metal",
    "punk", "reggae", "soul", "funk", "disco", "edm", "dubstep", "trap",
    "ambient music", "new age", "world music",
    "loop", "sample", "bpm",
    "banjo", "ukulele", "mandolin", "harp", "sitar", "harmonica",
    "accordion", "bagpipe", "didgeridoo",
    "xylophone", "marimba", "vibraphone", "glockenspiel", "tubular bells",
    "timpani", "bongo", "conga", "djembe", "tabla",
    "jingle", "bell", "chime", "gong",
    
    # Music genres/styles (additional)
    "soundtrack", "score", "composition", "arrangement",
    "verse", "chorus", "bridge", "intro", "outro",
    "acoustic", "electric", "unplugged",
}


def should_exclude(tags: list[str]) -> bool:
    """Check if any tag matches our exclusion list."""
    if not tags:
        return False
    
    tags_lower = {t.lower().strip() for t in tags}
    
    for tag in tags_lower:
        # Direct match
        if tag in EXCLUDED_TAGS:
            return True
        # Partial match - if any excluded term is contained in the tag
        for excluded in EXCLUDED_TAGS:
            if excluded in tag or tag in excluded:
                return True
    
    return False


def get_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """Get duration of audio from bytes."""
    try:
        import io
        with io.BytesIO(audio_bytes) as f:
            info = sf.info(f)
            return info.duration
    except Exception:
        return None


def load_and_preprocess_audio(audio_bytes: bytes, target_sr: int = SAMPLE_RATE, 
                               max_duration: float = CLIP_DURATION_SECONDS) -> Optional[np.ndarray]:
    """Load audio bytes, resample, and trim to max_duration."""
    try:
        import io
        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f, dtype='float32')
        
        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        
        # Trim to max_duration (first N seconds)
        max_samples = int(max_duration * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        # Pad if shorter than max_duration
        if len(waveform) < max_samples:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode='constant')
        
        return waveform.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Create ONNX inference session with GPU support."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers
    )
    
    # Check which provider is being used
    active_provider = session.get_providers()[0]
    print(f"ONNX Runtime using: {active_provider}")
    
    return session


def extract_embeddings_batch(session: ort.InferenceSession, 
                              waveforms: list[np.ndarray]) -> np.ndarray:
    """Extract embeddings for a batch of waveforms."""
    # Stack waveforms into batch
    batch = np.stack(waveforms, axis=0).astype(np.float32)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    embeddings = session.run([output_name], {input_name: batch})[0]
    
    return embeddings


def construct_freesound_url(username: str, sound_id: int) -> str:
    """Construct FreeSound URL from username and ID."""
    return f"https://freesound.org/people/{username}/sounds/{sound_id}/"


def process_dataset():
    """Main processing function."""
    print("=" * 60)
    print("FreeSound-LAION-640k Dataset Processor")
    print("=" * 60)
    
    # Load ONNX model
    print(f"\n1. Loading ONNX model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    session = create_onnx_session(MODEL_PATH)
    
    # Load dataset
    print("\n2. Loading dataset from HuggingFace...")
    print("   Dataset: benjamin-paine/freesound-laion-640k")
    
    dataset = load_dataset(
        "benjamin-paine/freesound-laion-640k",
        split="train",
        streaming=True  # Use streaming to avoid downloading all at once
    )
    
    # Process and filter
    print("\n3. Processing and filtering clips...")
    print(f"   - Max duration: {MAX_DURATION_SECONDS}s")
    print(f"   - Clip duration for embedding: {CLIP_DURATION_SECONDS}s")
    print(f"   - Filtering out speech, singing, and music")
    
    embeddings_data = []
    batch_waveforms = []
    batch_metadata = []
    
    stats = {
        "total_processed": 0,
        "filtered_duration": 0,
        "filtered_tags": 0,
        "failed_audio": 0,
        "successful": 0,
    }
    
    # Create output directory
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    progress = tqdm(dataset, desc="Processing clips")
    
    for item in progress:
        stats["total_processed"] += 1
        
        # Get metadata
        tags = item.get("tags", [])
        username = item.get("username", "")
        sound_id = item.get("id", 0)
        
        # Filter by tags
        if should_exclude(tags):
            stats["filtered_tags"] += 1
            continue
        
        # Get audio data
        audio_data = item.get("audio", {})
        if not audio_data:
            stats["failed_audio"] += 1
            continue
        
        # Check audio format - datasets library provides audio as dict with 'array' and 'sampling_rate'
        if isinstance(audio_data, dict):
            waveform = audio_data.get("array")
            sr = audio_data.get("sampling_rate", SAMPLE_RATE)
            
            if waveform is None:
                stats["failed_audio"] += 1
                continue
            
            waveform = np.array(waveform, dtype=np.float32)
            
            # Calculate duration
            duration = len(waveform) / sr
            
            # Filter by duration
            if duration > MAX_DURATION_SECONDS:
                stats["filtered_duration"] += 1
                continue
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Convert stereo to mono if needed
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Pad/trim to clip duration
            max_samples = int(CLIP_DURATION_SECONDS * SAMPLE_RATE)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode='constant')
        else:
            stats["failed_audio"] += 1
            continue
        
        # Add to batch
        batch_waveforms.append(waveform)
        batch_metadata.append({
            "username": username,
            "sound_id": sound_id,
        })
        
        # Process batch when full
        if len(batch_waveforms) >= BATCH_SIZE:
            try:
                embeddings = extract_embeddings_batch(session, batch_waveforms)
                
                for i, (emb, meta) in enumerate(zip(embeddings, batch_metadata)):
                    idx = stats["successful"] + 1
                    embeddings_data.append({
                        "id": f"{idx:012d}",
                        "embedding": emb.tolist(),
                        "freesound_url": construct_freesound_url(meta["username"], meta["sound_id"])
                    })
                    stats["successful"] += 1
                
            except Exception as e:
                print(f"\nError processing batch: {e}")
            
            batch_waveforms = []
            batch_metadata = []
        
        # Update progress
        progress.set_postfix({
            "kept": stats["successful"],
            "filtered": stats["filtered_tags"] + stats["filtered_duration"]
        })
        
        # Save periodically (every 10000 successful embeddings)
        if stats["successful"] > 0 and stats["successful"] % 10000 == 0:
            print(f"\n   Checkpoint: Saving {len(embeddings_data)} embeddings...")
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(embeddings_data, f)
    
    # Process remaining batch
    if batch_waveforms:
        try:
            embeddings = extract_embeddings_batch(session, batch_waveforms)
            
            for i, (emb, meta) in enumerate(zip(embeddings, batch_metadata)):
                idx = stats["successful"] + 1
                embeddings_data.append({
                    "id": f"{idx:012d}",
                    "embedding": emb.tolist(),
                    "freesound_url": construct_freesound_url(meta["username"], meta["sound_id"])
                })
                stats["successful"] += 1
        
        except Exception as e:
            print(f"\nError processing final batch: {e}")
    
    # Save final results
    print(f"\n4. Saving {len(embeddings_data)} embeddings to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f)
    
    # Print stats
    print("\n" + "=" * 60)
    print("Processing Statistics:")
    print("=" * 60)
    print(f"  Total processed:      {stats['total_processed']:,}")
    print(f"  Filtered (duration):  {stats['filtered_duration']:,}")
    print(f"  Filtered (tags):      {stats['filtered_tags']:,}")
    print(f"  Failed audio:         {stats['failed_audio']:,}")
    print(f"  Successful:           {stats['successful']:,}")
    print("=" * 60)
    
    return embeddings_data


def upload_to_pinecone(embeddings_data: Optional[list] = None):
    """Upload embeddings to Pinecone."""
    print("\n5. Uploading to Pinecone...")
    
    if embeddings_data is None:
        if not OUTPUT_JSON.exists():
            print(f"Error: {OUTPUT_JSON} not found. Run processing first.")
            return
        
        print(f"   Loading embeddings from {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
    
    print(f"   Total embeddings to upload: {len(embeddings_data):,}")
    
    # Get API key
    api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key: ")
    if not api_key:
        raise ValueError("Pinecone API Key is required")
    
    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    print(f"   Connected to index: {INDEX_NAME}")
    
    # Upsert in batches
    batch_size = 100
    print(f"   Uploading in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Uploading"):
        batch = embeddings_data[i:i + batch_size]
        
        vectors_to_upsert = [{
            "id": item["id"],
            "values": item["embedding"],
            "metadata": {
                "freesound_url": item["freesound_url"]
            }
        } for item in batch]
        
        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"\nError upserting batch {i // batch_size + 1}: {e}")
    
    print("\n   Upload complete!")
    print(index.describe_index_stats())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process FreeSound-LAION-640k dataset and upload to Pinecone"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip processing, only upload existing embeddings to Pinecone"
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process dataset, skip Pinecone upload"
    )
    
    args = parser.parse_args()
    
    if args.upload_only:
        upload_to_pinecone()
    elif args.process_only:
        process_dataset()
    else:
        embeddings_data = process_dataset()
        upload_to_pinecone(embeddings_data)


if __name__ == "__main__":
    main()
