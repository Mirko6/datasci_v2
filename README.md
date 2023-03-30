# Advanced Data Science Final Assignment

This repo is built on a Fynesse Template and provides a library for predicting house prices in the UK.

This repository has been built for a project for [DataSci course](https://mlatcl.github.io/advds/) at the University of Cambridge. The aim of the project was to learn a datascience pipeline by using Access, Assess, and Address division of work. In the Access part, the task was to set-up an AWS db, connect to it, load data to it and create some functions that communicate with it. In the Assess part, we analyzed the datasets that we were using. Finally, in the Address part we wrote a household price prediciton function and visualised the result.

This repository contains helper functions mainly for Access and Assess parts. I have used this repository in **[this google colab](https://colab.research.google.com/drive/1iQ3LvCCZqIeyjgNYgSKoCExcABQ1Wxvu?usp=sharing)** where I used it for predicting household prices in the UK.

Let me now describe the file structure

```bash
├── fynesse                 # The main folder
│   ├── access_external.py  # 1st part of access - responsible for fetching and uploading data
│   ├── access_db.py        # 2nd part of access - responsible for communicating with the db which already has the data
│   ├── assess.py           # functions designed to help us assess the data we have + get_training_data function
│   └── addresss.py         # functions designed to help us address the price prediction task
├── play.ipynb              # [not important]: notebook where I was uploading the data to database
└── modify_db.ipynb         # [not important]: notebook where I was modifying the db

```
