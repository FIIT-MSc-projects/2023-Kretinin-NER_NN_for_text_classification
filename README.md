# MSc-Kretinin-NER_NN_for_text_classification

Mykyta's Msc project FIIT STU Bratislava, NER and Topic modeling

## Registration number of work in the information system

FIIT-xxxx-xxxxxx

## MSc task

V dnešnej dobe sa dennodenne vytvára veľa textových dát ako knihy, novinové alebo vedecké články. Pre extrakciu cenných informácií je dôležitá identifikácia významných pojmov z textov napr. pomocou rozpoznávania pomenovaných entít (named entity recognition, NER). Táto úloha by sa mala vykonávať ako pivotná, pretože anotované zmienky zohrávajú dôležitú úlohu pri dolovaní textu. Tradičné NER prístupy sú vo veľkej miere závislé od rozsiahlych slovníkov, cieľových pravidiel alebo dobre zostavených korpusov. Tieto metódy sú momentálne nahradené prístupom založeným na hlbokom učení, ktoré sú menej závislé od ručne vyrobených prvkov.
Analyzujte súčasný stav problematiky v oblasti spracovania a klasifikácie textu s technikami zameranými na NER s neurónovými sieťami. Navrhnite a implementujte efektívnu metódu na vytvorenie a použitie inteligentného modelu pre analýzu a klasifikáciu textových dát s použitím NER. Cieľom je vytvorenie kombinovaného prístupu NER s inými technikami spracovania prirodzeného jazyka na zlepšenie automatického procesu triedenia textových dokumentov vo vybraných štúdiách. Vyhodnoťte navrhovaný prístup a jeho výstupy pomocou dostupných metrík. Porovnajte dosiahnuté výsledky s inými existujúcimi riešeniami.

## Project Organization
```
├── README.md           <- The top-level README for developers using this project.
├── data
│   ├── external        <- Data from third-party sources.
│   ├── corpora         <- Processed corpora ready for use.
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump.
├── kaggle              <- Kaggle datasets used to train NER models
│                           
├── models              <- Trained and serialized models 
│                           (folders with .pickle)
│                           
├── notebooks           <- Notebooks with source 
│                           code of experiments
│                                                 
├── requirements.txt    <- The requirements file for 
│                           reproducing the analysis environment, e.g.
│                           generated with `pipreqs`
```

## User Manual

All source code used in the experiments is located in /notebooks folder, in the .ipynb . 
It is advised to create an environment for experiments and install a CUDA toolkit to train and test tensorflow models faster.
All requirements can be installed using `pip install -r requirements.txt`