# Confronto Recognition

## Ultimi test

Questa sezione viene sovrascritta a ogni nuova campagna di test automatica.

### Obiettivo

Rifare il run della recognition dopo l'aggiunta di nuove immagini nel dataset custom.

Configurazione lanciata:
- train mode: `all`
- eval mode: `custom`

### Risultati

- training
  - custom gallery identities: `2`
  - custom gallery samples: `60`
  - ORL gallery samples: `200`
  - LFW gallery samples: `3528`
  - total faces: `3788`
  - total identities: `943`
  - gallery loading time: `1.46s`
  - LBPH training time: `14.46s`
  - total training time: `27.17s`

- evaluation su `custom`
  - probe identities: `2`
  - probe samples: `20`
  - correct: `17`
  - accuracy: `0.85`
  - genuine matches: `17`
  - impostor matches: `3`
  - mean genuine confidence: `69.6402`
  - mean impostor confidence: `91.1955`
  - suggested threshold: `94.7661`
  - evaluation time: `4.18s`
  - total evaluation time: `14.99s`
  - total experiment time: `42.25s`

### Lettura

- il custom aggiornato e migliorato rispetto al run precedente su custom
- accuracy su custom: `0.70 -> 0.85`
- il dataset custom ora contiene `2` identita e `60` immagini gallery, quindi il test e piu credibile del sanity check iniziale
- il modello allenato su `all` continua a usare bene ORL e LFW come supporto, ma sul custom aggiornato il risultato e decisamente migliore

## Setup corrente

- trainer: `src/recognition/recognition_model_trainer.py`
- evaluator: `src/recognition/recognition_model_evaluator.py`
- runner: `src/recognition/recognition_experiment_runner.py`
- modello salvato in:
  - `models/lbph_model.xml`
  - `models/label_map.pkl`
  - `models/recognition_config.json`

## Dataset disponibili

- `custom`
- `orl`
- `lfw`
- `custom_orl`
- `all`

Nota sui mode:
- `custom_orl` = mix tra `custom` e `orl`, pensato come training realistico senza `LFW`
- `all` = esperimento completo con `custom + orl + lfw`

Nota sulla threshold runtime:
- la threshold ufficiale del runtime non viene piu aggiornata da run troppo facili
- un run puo aggiornare `models/recognition_config.json` solo se:
  - appartiene a una coppia dichiarata valida
  - contiene sia `genuine` sia `impostor`
  - usa una regola non-fallback
- le coppie valide correnti sono:
  - `orl -> orl`
  - `custom_orl -> orl`
- config runtime attualmente salvato:
  - source: `custom_orl_to_orl`
  - threshold: `89.9169`
  - threshold rule: `midpoint_max_genuine_min_impostor`

## Struttura recognition

- `data/recognition/custom/gallery`
- `data/recognition/custom/probe`
- `data/recognition/orl/gallery`
- `data/recognition/orl/probe`
- `data/recognition/lfw/gallery`
- `data/recognition/lfw/probe`

## Roadmap operativa

Stato corrente:
- pipeline recognition: `OK`
- LBPH: `stabile ma ancora migliorabile`
- prossimo step attivo: `fare tuning piccolo ma serio dei parametri LBPH`

### Obiettivo

Portare la recognition alla miglior versione onesta e difendibile del progetto, migliorando coerenza, misurazione e robustezza senza inseguire risultati irrealistici su LFW.

### Regole fisse

- `LFW` resta dataset di stress test e limite del metodo
- i confronti principali da usare sono:
  - `custom -> custom`
  - `orl -> orl`
  - `custom_orl -> custom`
  - `custom_orl -> orl`
- `ORL` resta dataset di supporto per verificare che la pipeline continui a comportarsi bene
- non cambiare insieme threshold, preprocessing, parametri LBPH e protocollo nello stesso test
- i risultati vanno sempre confrontati con lo storico gia presente in questo file

### Priorita massima

- [x] rendere piu robusta la regola di threshold
  - obiettivo: sostituire `max(genuine) + 5` con una regola meno fragile agli outlier
  - candidati da testare:
    - midpoint tra `max(genuine)` e `min(impostor)`
    - `mean(genuine) + k * std(genuine)`
  - baseline di riferimento:
    - `all -> custom`
  - file coinvolti:
    - `src/recognition/recognition_model_evaluator.py`
    - `models/recognition_config.json`

- [x] aggiungere una guardia runtime se il preprocessing recognition fallisce
  - obiettivo: evitare crash quando `preprocess_face_for_recognition(...)` restituisce `None`
  - comportamento atteso:
    - saltare la recognition
    - mostrare `Unknown` o bloccare il riconoscimento in modo sicuro
  - file coinvolti:
    - `src/pipeline/run_pipeline_full.py`
    - `src/preprocessing/face_preprocessing.py`

- [x] separare meglio il protocollo di training da quello di stress test
  - obiettivo: evitare che `LFW` sporchi il training del caso reale
  - decisione da verificare:
    - `custom_orl` come mix realistico principale
    - `all` mantenuto come esperimento completo con `LFW`
    - `LFW` trattato come stress test e non come default concettuale di training
  - nuova matrice di mode:
    - training: `custom`, `orl`, `lfw`, `custom_orl`, `all`
    - evaluation: `custom`, `orl`, `lfw`, `custom_orl`, `all`
  - file coinvolti:
    - `src/recognition/recognition_dataset_loader.py`
    - `src/recognition/recognition_experiment_runner.py`
    - `recognition.md`

### Priorita alta

- [x] allineare la threshold tra evaluator e runtime
  - obiettivo: usare a runtime una soglia coerente con quella stimata in evaluation
  - file coinvolti:
    - `src/recognition/recognition_model_evaluator.py`
    - `src/recognition/face_recognizer.py`
    - `src/pipeline/run_pipeline_full.py`
  - nota aggiornata:
    - la threshold runtime viene salvata solo da run con genuine e impostor reali
    - i run con fallback `max_genuine_plus_5` non possono piu aggiornare il config

- [x] rendere l'evaluator piu informativo
  - obiettivo: capire se gli errori dipendono da identita, threshold o dataset
  - aggiunte minime utili:
    - accuracy per identita
    - identita piu confuse
    - statistiche `min / max / std` di genuine e impostor scores
    - conteggio `Unknown` con threshold candidate
  - file coinvolti:
    - `src/recognition/recognition_model_evaluator.py`

- [x] confrontare in modo pulito i training utili
  - obiettivo: capire quale training ha davvero senso per il target del progetto
  - confronti desiderati:
    - `custom -> custom`
    - `custom_orl -> custom`
    - `orl -> orl`
    - `custom_orl -> orl`
    - `all -> custom` come confronto completo con `LFW`
  - file coinvolti:
    - `src/recognition/recognition_experiment_runner.py`
    - `recognition.md`

### Priorita media

- [ ] fare tuning piccolo ma serio dei parametri LBPH
  - obiettivo: testare se una configurazione migliore aiuta davvero sul custom
  - griglia iniziale consigliata:
    - `radius`: `1, 2, 3`
    - `neighbors`: `8, 16`
    - `grid_x`: `6, 8, 10`
    - `grid_y`: `6, 8, 10`
  - file coinvolti:
    - `src/recognition/recognition_model_trainer.py`
    - `src/recognition/recognition_experiment_runner.py`
  - stato implementazione:
    - trainer e runner ora supportano parametri LBPH configurabili
    - i test di tuning non sono ancora stati lanciati

- [ ] testare 1-2 varianti semplici di preprocessing
  - obiettivo: aumentare stabilita del crop senza stravolgere la pipeline
  - varianti sensate:
    - padding leggero del bounding box
    - controllo minimo sulla dimensione del volto
    - confronto `equalizeHist ON/OFF`
    - confronto `equalizeHist` vs `CLAHE`
  - file coinvolti:
    - `src/preprocessing/face_preprocessing.py`

### Priorita bassa

- [ ] lasciare `recognition_dataset_loader.py` quasi invariato
  - obiettivo: evitare refactor estetici inutili sul loader
  - modifiche sensate solo se servono ai test:
    - statistiche per identita
    - supporto a test rapidi o report piu leggibili
  - file coinvolti:
    - `src/recognition/recognition_dataset_loader.py`

### Ordine consigliato

1. rendere piu robusta la threshold runtime
2. aggiungere guardia su `face is None` nel runtime
3. chiarire il ruolo di `LFW` e separare training vs stress test
4. arricchire l'evaluator
5. confrontare i training davvero utili
6. fare tuning LBPH piccolo
7. testare preprocessing semplice
8. lasciare `LFW` come limite dichiarato del metodo

### Criteri di confronto

Per ogni modifica confrontare sempre:
- accuracy
- genuine matches
- impostor matches
- mean genuine confidence
- mean impostor confidence
- suggested threshold
- tempo totale di training
- tempo totale di evaluation

### Nota metodologica

- l'obiettivo non e rendere `LBPH` competitivo su `LFW`
- l'obiettivo e chiudere le incoerenze del modulo, misurare meglio e ottenere la miglior versione realistica della recognition per il dataset custom
- `custom -> custom` serve come sanity check
- `orl -> orl` e `custom_orl -> orl` sono i run di riferimento piu difendibili per la threshold runtime
- `all -> custom` resta utile come confronto completo, ma non fa parte dei test base e non deve guidare la threshold ufficiale
- ogni nuovo test va aggiunto sotto `Ultimi test` o `Storico test`, senza sostituire questa roadmap

## Tuning LBPH

### Criterio di scelta

- la configurazione migliore non e quella con accuracy massima in assoluto
- prima si scartano le configurazioni che peggiorano troppo `custom_orl -> custom`
- tra quelle rimaste si preferisce quella che su `custom_orl -> orl` riduce meglio gli `accepted_impostor` alla `suggested_threshold`
- a parita di `accepted_impostor`, si preferisce quella con meno `rejected_genuine`
- se i risultati restano molto vicini, si considera come ulteriore criterio l'accuracy su `orl`

### Classifica provvisoria

Questa classifica e basata sui test gia lanciati con `neighbors = 8`. I test con `neighbors = 16` sono ancora da completare.

1. `radius=1`, `neighbors=8`, `grid_x=6`, `grid_y=6`
   - `custom_orl -> orl`: accuracy `0.9600`, `accepted_impostor=4`, `rejected_genuine=5`
   - `custom_orl -> custom`: accuracy `1.0000`
2. `radius=2`, `neighbors=8`, `grid_x=6`, `grid_y=6`
   - `custom_orl -> orl`: accuracy `0.9700`, `accepted_impostor=4`, `rejected_genuine=11`
   - `custom_orl -> custom`: accuracy `1.0000`
3. `radius=1`, `neighbors=8`, `grid_x=8`, `grid_y=8`
   - `custom_orl -> orl`: accuracy `0.9200`, `accepted_impostor=8`, `rejected_genuine=7`
   - `custom_orl -> custom`: accuracy `1.0000`
4. `radius=2`, `neighbors=8`, `grid_x=8`, `grid_y=8`
   - `custom_orl -> orl`: accuracy `0.9400`, `accepted_impostor=4`, `rejected_genuine=8`
   - `custom_orl -> custom`: accuracy `1.0000`
5. `radius=1`, `neighbors=8`, `grid_x=10`, `grid_y=10`
   - `custom_orl -> orl`: accuracy `0.9200`, `accepted_impostor=8`, `rejected_genuine=12`
   - `custom_orl -> custom`: accuracy `1.0000`
6. `radius=2`, `neighbors=8`, `grid_x=10`, `grid_y=10`
   - `custom_orl -> orl`: accuracy `0.9000`, `accepted_impostor=7`, `rejected_genuine=13`
   - `custom_orl -> custom`: accuracy `0.9500`

### Lettura provvisoria

- `grid_x = grid_y = 6` e al momento la scelta piu promettente
- tra le configurazioni gia provate, `radius=1`, `neighbors=8`, `grid=6` e la migliore in termini di equilibrio tra impostor e genuine su `orl`
- `radius=2`, `neighbors=8`, `grid=6` resta una candidata forte, ma perde punti per il numero maggiore di `rejected_genuine`
- le configurazioni con `grid=10` stanno peggiorando sia la separazione su `orl` sia la stabilita sul `custom`
- prima di chiudere il tuning vanno ancora verificati i casi con `neighbors = 16`

## Confronti fissi correnti

Questi sono i confronti minimi da rifare quando cambia qualcosa nella recognition, lanciati in sequenza per evitare sovrascritture di `lbph_model.xml` e `recognition_config.json`.

### `custom -> custom`

- gallery samples: `60`
- identities: `2`
- probe samples: `20`
- correct: `20`
- accuracy: `1.0000`
- genuine matches: `20`
- impostor matches: `0`
- mean genuine confidence: `73.3075`
- suggested threshold: `104.4171`
- suggested threshold rule: `max_genuine_plus_5_fallback`
- total experiment time: `0.54s`
- runtime threshold update: `no`

### `custom_orl -> custom`

- gallery samples: `260`
- identities: `42`
- probe samples: `20`
- correct: `20`
- accuracy: `1.0000`
- genuine matches: `20`
- impostor matches: `0`
- mean genuine confidence: `73.3075`
- suggested threshold: `104.4171`
- suggested threshold rule: `max_genuine_plus_5_fallback`
- total experiment time: `1.92s`
- runtime threshold update: `no`

### `orl -> orl`

- gallery samples: `200`
- identities: `40`
- probe samples: `200`
- correct: `184`
- accuracy: `0.9200`
- genuine matches: `184`
- impostor matches: `16`
- mean genuine confidence: `73.1234`
- mean impostor confidence: `88.7856`
- suggested threshold: `89.9169`
- suggested threshold rule: `midpoint_max_genuine_min_impostor`
- total experiment time: `3.74s`
- runtime threshold update: `yes`

### `custom_orl -> orl`

- gallery samples: `260`
- identities: `42`
- probe samples: `200`
- correct: `184`
- accuracy: `0.9200`
- genuine matches: `184`
- impostor matches: `16`
- mean genuine confidence: `73.1234`
- mean impostor confidence: `88.7856`
- suggested threshold: `89.9169`
- suggested threshold rule: `midpoint_max_genuine_min_impostor`
- total experiment time: `4.69s`
- runtime threshold update: `yes`

### Lettura dei confronti

- `custom` e `custom_orl` sono equivalenti su `custom`
- `orl` e `custom_orl` sono equivalenti su `orl`
- quindi `custom_orl` e al momento il candidato piu pulito come training misto realistico
- per la threshold runtime, i run di riferimento sono `orl -> orl` e `custom_orl -> orl`

## Confronti extra

Questi run restano utili come confronto allargato o stress experiment, ma non fanno parte dei test base e non devono guidare la threshold ufficiale del runtime.

### `all -> custom`

- gallery samples: `3788`
- identities: `943`
- probe samples: `20`
- correct: `17`
- accuracy: `0.8500`
- genuine matches: `17`
- impostor matches: `3`
- mean genuine confidence: `69.6402`
- mean impostor confidence: `91.1955`
- suggested threshold: `88.9671`
- suggested threshold rule: `midpoint_max_genuine_min_impostor`
- total experiment time: `42.81s`

## Storico test

### Run `all -> all`

- `custom`
  - probe samples: `10`
  - correct: `7`
  - accuracy: `0.7000`
  - genuine matches: `7`
  - impostor matches: `3`
  - mean genuine confidence: `79.7125`
  - mean impostor confidence: `91.1955`
  - suggested threshold: `94.7661`
  - eval time: `2.08s`

- `orl`
  - probe samples: `200`
  - correct: `184`
  - accuracy: `0.9200`
  - genuine matches: `184`
  - impostor matches: `16`
  - mean genuine confidence: `73.1234`
  - mean impostor confidence: `88.7600`
  - suggested threshold: `104.5012`
  - eval time: `46.53s`

- `lfw`
  - probe samples: `4078`
  - correct: `282`
  - accuracy: `0.0692`
  - genuine matches: `282`
  - impostor matches: `3796`
  - mean genuine confidence: `49.4931`
  - mean impostor confidence: `53.2856`
  - suggested threshold: `80.4652`
  - eval time: `917.25s`

### Dettagli training storico `all`

- custom gallery samples: `30`
- ORL gallery samples: `200`
- LFW gallery samples: `3528`
- total faces: `3758`
- total identities: `942`
- gallery loading time: `1.20s`
- LBPH training time: `14.61s`
- total training time: `27.22s`

### Lettura corrente

- la pipeline nuova della recognition funziona end-to-end
- ORL continua ad andare bene
- LFW resta il dataset piu difficile per LBPH
- il custom aggiornato e migliorato sensibilmente rispetto al primo run

## Prossimi test utili

- `custom -> custom`
- `orl -> orl`
- `lfw -> lfw`
- `all -> orl`
- `all -> lfw`
