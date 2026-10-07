# data/

Git does not track the files in this folder, except this README. Do not commit audio or
transcripts.

## Earnings-21

| Item | Value |
|---|---|
| Name | Earnings-21 (Rev.com speech-datasets) |
| URL | <https://github.com/revdotcom/speech-datasets> |
| License | CC BY-SA 4.0. Read the license in the repository before you use the data |
| Content | 44 public earnings calls (MP3), reference transcripts as `.nlp` token files with speaker labels |
| Size | 8 MB to 80 MB for each MP3. callscribe decodes in memory and writes no WAV copy |

Download only the `earnings21` folder:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/revdotcom/speech-datasets.git data/speech-datasets
cd data/speech-datasets
git sparse-checkout set earnings21
cd ../..
callscribe manifest --root data/speech-datasets/earnings21 --out data/earnings21.json
```

`callscribe manifest` pairs each audio file with the `.nlp` file of the same name. If an `.rttm`
file of the same name exists anywhere under the root, the manifest adds it, and the evaluation
then also gives DER. Without RTTM files, the evaluation gives WER, CER and cpWER only.
Earnings-22 in the same repository has the same layout and is a larger set.

## Expected `.nlp` columns

`token|speaker|ts|endTs|punctuation|case|tags|wer_tags`. callscribe reads `token`, `speaker`,
`ts` and `endTs`.

## No download

`callscribe synth --out data/synthetic` writes synthetic calls (WAV), exact `.nlp` and `.rttm`
references and `manifest.json`. The tests use only synthetic calls.
