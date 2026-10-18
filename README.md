# OUME for Anki

OUME replaces the next-interval calculation for one deck. Anki still shows the cards and the Again / Hard / Good / Easy buttons. FSRS does not choose those intervals.

The study screen is Anki's normal reviewer, not a separate window.

## What changes

Before the answer buttons are drawn, each button's delay is rewritten from the OUME-Pragmatic transition:

- a same-day learning pulse stays under one day
- a graduated card is scheduled in whole days
- pressing the button records that same transition, so the next review does not fall back to FSRS

Set `deck_name` in the add-on config. An empty name applies OUME to every deck. Other decks are left on Anki's scheduler.

## Install

Download `oume_study.ankiaddon` from the GitHub Release. In desktop Anki use Tools → Add-ons → Install from file, then restart. Open Tools → OUME 调度状态 to confirm it is recording reviews.

A phone can download that file and transfer it to the computer. AnkiDroid does not run this desktop add-on, so the scheduler change applies to desktop Anki.

## Development

```bash
PYTHONPATH=. python3 -m unittest tests.test_schedule
python3 -m py_compile __init__.py oume/*.py
```
