# OUME for Anki

Desktop add-on that runs the OUME-Pragmatic scheduler on one selected deck. Other decks keep Anki's scheduler. The numerical transitions are vendored in `oume/scheduler_core.py` (reference implementation v1.16).

This window is an Anki desktop (aqt/Qt) dialog. AnkiDroid and AnkiMobile do not run it. The `.ankiaddon` file can still be downloaded on a phone and installed on desktop Anki.

## What a study session does

1. New cards get three same-day contacts: recognition, spelling, then generation, with a 10 minute gap.
2. The first recognition contact is a guided look, not a scored recall.
3. If the gap is at least 2 minutes and no review is due, the session tells you to leave and come back.
4. Cross-day reviews use a normalized prediction-error stability update.
5. A one-parameter memory-scale estimate is updated from those reviews and ignored until its uncertainty is low.
6. Hints and wrong answers are Again. Multiple choice can never be Easy.

The add-on does not claim a measured learning benefit. That requires a preregistered human trial.

## Install on desktop

Anki 26 reads add-ons from `~/.local/share/Anki2/addons21` (macOS: `~/Library/Application Support/Anki2/addons21`, Windows: `%APPDATA%\Anki2\addons21`).

1. Download `oume_study.ankiaddon` from the GitHub Release.
2. In Anki: Tools → Add-ons → Install from file, and choose that file. Or double-click it while Anki is closed.
3. Restart Anki. The folder name must stay `oume_study`. Do not rename it to `oume`, or Anki will import this add-on instead of the scheduler package.

From a phone: open the Release page, download `oume_study.ankiaddon`, then transfer it to the computer (USB, cloud, or mail) and install it with the steps above. AnkiDroid can import the file, but this Qt dialog will not appear there.

## Use

Open **Tools → OUME Study**. Choose the deck, set the session budget, and start. Answers are stored in `~/.local/share/oume/anki-profile.json`, separate from Anki's FSRS state, so the two schedulers do not both advance the same card.

## Development

```bash
PYTHONPATH=. python3 -c "import oume; from oume.session import StudyEngine"
python3 -m py_compile __init__.py oume/*.py
```
