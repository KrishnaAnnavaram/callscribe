# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **call** | One audio recording of an earnings call | file (for the audio), meeting |
| **transcript** | The words, turns, utterances and metadata of one call | output, result (for a transcript) |
| **word** | One recognised token with a start time, an end time and a speaker | token (outside the `.nlp` format) |
| **turn** | One time span in which one speaker talks, from the diarizer | segment, region (for a speaker span) |
| **speech region** | One time span that the VAD marks as speech | turn (for a VAD span) |
| **utterance** | Consecutive words of one speaker with no long pause | sentence, paragraph |
| **speaker label** | The name of a speaker, for example `SPEAKER_00` | speaker ID, voice |
| **ASR backend** | `scripted` or `faster-whisper` | recogniser, engine |
| **diarizer** | `spectral`, `pyannote` or `reference` | speaker model, clusterer |
| **scripted ASR** | The offline simulator that returns reference words with seeded errors | fake ASR, mock |
| **chunk** | One audio window that the ASR receives, with its own region | slice, piece |
| **own region** | The part of a chunk that keeps the words whose midpoint is in it | core, center |
| **reference** | The true words, speakers or turns of a call | ground truth, gold |
| **manifest** | The JSON list of calls with audio, `.nlp`, `.rttm` and split | index, catalog |
| **dev split** | The calls that tuning uses | validation set |
| **test split** | The calls that the evaluation reports | holdout |
| **WER** | Word error rate after normalisation | accuracy, match percentage |
| **cpWER** | WER of the speaker streams after the best speaker mapping | speaker WER |
| **DER** | Diarization error rate: missed speech, false alarm and confusion, divided by reference speech | diarization accuracy |
| **collar** | The time around each reference boundary that DER does not score | margin, tolerance |
| **pooled** | Summed over all calls before the division | average (for a pooled value) |
| **token** | The Hugging Face access token in `HF_TOKEN` | key, password |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **decode** | Change an audio file into 16 kHz mono samples in memory |
| **detect** | Find the speech regions with the VAD |
| **transcribe** | Change audio into words with times |
| **diarize** | Change audio into turns with speaker labels |
| **align** | Give each word the speaker label of the turn that overlaps it most |
| **merge** | Keep each word of overlapping chunks only once |
| **tune** | Find the diarizer parameters with the lowest pooled DER on the dev split |
| **score** | Calculate WER, CER, cpWER and DER for one call |
| **evaluate** | Run the pipeline on a manifest split and score each call |
| **export** | Write a transcript as JSON, SRT, TXT or RTTM |
