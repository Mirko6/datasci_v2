# Fynesse Template

This repo is built on a Fynesse Template and provides a library for predicting house
prices in the UK.

Let me now describe the files that I have added to this repository

```bash
├── fynesse                 # The main folder
│   ├── access_external.py  # 1st part of access - responsible for fetching and uploading data
│   ├── access_db.py        # 2nd part of access - responsible for communicating with the db which already has the data
│   ├── assess.py           # functions designed to help us assess the data we have + get_training_data function
│   └── addresss.py         # functions designed to help us address the price prediction task
├── play.ipynb              # [not important]: notebook where I was uploading the data to database
└── modify_db.ipynb         # [not important]: notebook where I was modifying the db

```
