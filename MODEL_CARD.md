# ThatSoundsLikeMe ONNX embedding model

## Summary

`model_v1.onnx` is the project-owned audio embedding model used by
ThatSoundsLikeMe. It converts a short vocal imitation into a 960-value vector
for similarity search. The same artifact is bundled as `web/public/model.onnx`
in the public [`thatsoundslikeme/app`](https://github.com/thatsoundslikeme/app)
repository and runs locally through ONNX Runtime Web.

- **Authors:** Christos Plachouras, Aditya Bhattacharjee and Sungkyun Chang
- **Licence:** MIT, under this repository's `LICENSE`
- **SHA-256:** `a652b09c76c754fe32085d85d90824e2842b57badc3aa4cab66eb8ba9312d4d6`
- **Output:** 960 finite floating-point values

The model was developed from the team's first-place entry in the [Audio
Engineering Society (AES)](https://aes2.org/) [AIMLA Querying by Vocal
Imitation Challenge 2025](https://qvim-aes.github.io/#results). The desktop/web adaptation was
subsequently used in the project that won [Best app for Muse Hub and the Other
challenge at the 2025 London Music Technology
Hackathon](https://devpost.com/software/imitune).

## Intended use

The model is intended for query-by-vocal-imitation retrieval: a user makes a
short non-speech imitation of a desired sound and the embedding is compared
with embeddings of real-world sounds. It is not a speech-recognition,
speaker-identification, authentication or safety-critical model.

Microphone audio and inference stay on the user's device during ordinary
search. The application sends the embedding, not the raw search recording, to
its similarity-search API. A recording leaves the device only through the
separate, opt-in research feedback flow described by the application's privacy
policy and ethics documents.

## Limitations

Retrieval quality varies with the imitated sound, recording conditions,
microphone and indexed corpus. Similarity is not a guarantee that a returned
sound matches the user's intent, which is why results are presented for human
listening and optional rating. The model must not be used to infer identity or
sensitive characteristics from a voice.

## Artifact verification

From the repository root:

```sh
shasum -a 256 model_v1.onnx
```

The output must match the SHA-256 value above before the artifact is copied
into a release build.
