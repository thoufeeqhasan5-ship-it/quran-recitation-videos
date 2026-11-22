# Qur'an – The Recitation — video maker

Turns a topic file into a finished 9:16 English short: a calm English AI narrator, the real Arabic
ayah with its recitation and translation, nature photos that move like camera shots, fact cards,
captions, the page branding and an end card. Runs free on GitHub Actions, like karunaada-videos.

## What a video looks like
1. **Hook** (first ~3 s): a big line, e.g. "Swallowed by a giant fish", with "A story from the Qur'an".
2. **The story**, told by the narrator, with captions and short fact cards.
3. **The ayah**: a panel shows the Arabic, the surah and ayah number and the English translation.
   The reciter recites the Arabic first, then the narrator reads the translation.
4. **Lesson + call to action**, then the end card.

The Arabic, the translation and the recitation are **downloaded from api.alquran.cloud each time**,
so the verse is never typed by hand. No background music by default.

## One-time setup
1. Create a new **public** GitHub repo named `quran-recitation-videos`.
2. Upload everything in this folder, keeping the folders (`assets/`, `scripts/`, `topics/`).
   `.github` is hidden: on GitHub use *Add file → Create new file*, type
   `.github/workflows/make-video.yml` as the name and paste the file's contents.
   Do the same for `.github/workflows/download-all-videos.yml`.
3. Optional but recommended: add your Pixabay key (the same one as Karunaada) for better photos:
   *Settings → Secrets and variables → Actions → New repository secret*, name `PIXABAY_KEY`.
   No Hugging Face token is needed.

## Make a video
Actions → **Make Quran video** → Run workflow → topic `01` → Run.
Choices: narrator voice, recitation on/off, reciter, translation.
The first run takes ~10–15 min (downloads the voice model); later runs are faster.
When it finishes, open the run and download the zip under **Artifacts**:
- `video.mp4` – the finished short
- `voice.wav` – the narration and recitation alone
- `caption.txt` – caption and hashtags to paste (with the ayah reference)
- `credits.txt` – photos, translation and reciter used

## Voices
`bm_george` (British male, default), `bm_lewis`, `am_michael`, `am_adam`, `bf_emma`, `af_heart`.
Kokoro-82M, Apache-2.0. Speed can be changed with `VOICE_SPEED` in the workflow (default 0.92).

## Adding a topic
Copy `topics/01.json` to a new number and edit:
- `hook` – the big opening line.
- `phrases` – in speaking order. Story lines: `text` (captions + voice), optional `card`,
  `shot` (which photo), `ref` (the ayat it is based on, for your own checking; not shown).
  To change how a word is spoken without changing the caption, add `say`.
- The ayah line: `{"ayah": "21:87", "shot": 4, "surah": "Al-Anbiya"}` – only the reference.
  Add `"recite": false` to skip the recitation for that ayah.
- `shots` – photo search words, best first. Nature, sky, sea, desert, light. Never people.
- `caption` – post caption.

## Check before posting
- **Read every story line against its `ref` ayat.** The stories stay with what the Qur'an says,
  but please verify each one, or have someone knowledgeable check it, before posting.
- Look at each photo. People, statues and religious images are filtered out by their tags,
  but tags are not perfect. Change the search words in `shots` if a photo is wrong.
- Listen to names (Yunus, Zakariya, Judi…). If one sounds wrong, add a `say` with a spelling that
  sounds right.

## Local layout test (no internet)
`python scripts/make_video.py --topic 01 --test` – placeholder tone, photos and ayah.
