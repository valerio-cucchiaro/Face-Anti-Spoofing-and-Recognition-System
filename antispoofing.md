# Confronto PAD Runtime

## Ultimi test

Questa sezione viene sovrascritta a ogni nuova campagna di test automatica.

### Obiettivo

Capire:
- se il bilanciamento pre-estrazione di CASIA aiuta davvero nel setup `fast`
- come cambia il comportamento runtime aumentando il peso del dataset runtime

Regola di confronto usata per A vs B:
- recall in aumento
- FN in diminuzione
- FP non troppo in aumento
- precision accettabile
- F1 non peggiore

### Configurazioni lanciate

- `fast_A_r1`
  - train mode: `combined_runtime`
  - eval mode: `runtime`
  - speed: `fast`
  - CASIA ratio: `0.10`
  - Replay ratio: `0.10`
  - CASIA pre-balance: `False`
  - Replay pre-balance: `True`
  - runtime repeat: `1`

- `fast_B_r1`
  - train mode: `combined_runtime`
  - eval mode: `runtime`
  - speed: `fast`
  - CASIA ratio: `0.10`
  - Replay ratio: `0.10`
  - CASIA pre-balance: `True`
  - Replay pre-balance: `True`
  - runtime repeat: `1`

- `fast_A_r2`
  - stesso setup di `fast_A_r1`
  - runtime repeat: `2`

- `fast_A_r5`
  - stesso setup di `fast_A_r1`
  - runtime repeat: `5`

### Risultati

- `fast_A_r1`
  - train samples: `5989`
  - CASIA weight: `83.19%`
  - Replay weight: `14.78%`
  - Runtime weight: `2.04%`
  - accuracy: `0.832`
  - precision: `0.9524`
  - recall: `0.500`
  - F1: `0.6557`
  - confusion matrix: `[[84, 1], [20, 20]]`
  - live corretti: `20/40`
  - spoof corretti: `84/85`
  - total time: `3399.95s`

- `fast_B_r1`
  - train samples: `4212`
  - CASIA weight: `76.09%`
  - Replay weight: `21.01%`
  - Runtime weight: `2.90%`
  - accuracy: `0.792`
  - precision: `0.9375`
  - recall: `0.375`
  - F1: `0.5357`
  - confusion matrix: `[[84, 1], [25, 15]]`
  - live corretti: `15/40`
  - spoof corretti: `84/85`
  - total time: `2409.06s`

- `fast_A_r2`
  - train samples: `6111`
  - CASIA weight: `81.53%`
  - Replay weight: `14.48%`
  - Runtime weight: `3.99%`
  - accuracy: `0.824`
  - precision: `0.9500`
  - recall: `0.475`
  - F1: `0.6333`
  - confusion matrix: `[[84, 1], [21, 19]]`
  - live corretti: `19/40`
  - spoof corretti: `84/85`
  - total time: `3452.29s`

- `fast_A_r5`
  - train samples: `6477`
  - CASIA weight: `76.92%`
  - Replay weight: `13.66%`
  - Runtime weight: `9.42%`
  - accuracy: `0.824`
  - precision: `0.9500`
  - recall: `0.475`
  - F1: `0.6333`
  - confusion matrix: `[[84, 1], [21, 19]]`
  - live corretti: `19/40`
  - spoof corretti: `84/85`
  - total time: `3259.88s`

### Lettura

- tra A e B vince chiaramente `fast_A_r1`
- A ha recall piu alto (`0.500` vs `0.375`)
- A ha meno FN (`20` vs `25`)
- gli FP restano identici (`1`)
- la precision resta molto vicina (`0.9524` vs `0.9375`)
- anche F1 e accuracy sono migliori in A

- aumentando il runtime weight con `x2` e `x5` il risultato non migliora
- `x2` e `x5` peggiorano rispetto ad A base:
  - recall: `0.500 -> 0.475`
  - F1: `0.6557 -> 0.6333`
  - live corretti: `20 -> 19`
- tra `x2` e `x5` non c'e differenza pratica sui risultati runtime

- miglior setup di questa campagna:
  - `fast_A_r1`
  - quindi `fast`, CASIA non bilanciato, Replay bilanciato, `runtime_repeat = 1`

## Test completi

### Setup fisso

- train mode: `combined_runtime`
- eval mode: `runtime`
- experiment speed: `full`
- CASIA sample ratio: `1.00`
- Replay sample ratio: `1.00`
- CASIA pre-extraction balance: `False`
- Replay pre-extraction balance: `True`
- runtime repeat: `10`

### Ablation feature modes

- `full` baseline
  - X shape: `128`
  - accuracy: `0.776`
  - precision: `0.8750`
  - recall: `0.350`
  - F1: `0.5000`
  - confusion matrix: `[[83, 2], [26, 14]]`

- `lbp_only`
  - X shape: `108`
  - accuracy: `0.712`
  - precision: `0.8333`
  - recall: `0.125`
  - F1: `0.2174`
  - confusion matrix: `[[84, 1], [35, 5]]`

- `lbp_lft`
  - X shape: `116`
  - accuracy: `0.728`
  - precision: `0.8750`
  - recall: `0.175`
  - F1: `0.2917`
  - confusion matrix: `[[84, 1], [33, 7]]`

- `no_down_quality`
  - X shape: `122`
  - accuracy: `0.768`
  - precision: `0.9231`
  - recall: `0.300`
  - F1: `0.4528`
  - confusion matrix: `[[84, 1], [28, 12]]`

### Confronto TGSD

- `full` baseline
  - X shape: `128`
  - accuracy: `0.776`
  - precision: `0.8750`
  - recall: `0.350`
  - F1: `0.5000`

- `full + TGSD v1`
  - X shape: `130`
  - accuracy: `0.792`
  - precision: `0.8889`
  - recall: `0.400`
  - F1: `0.5517`

- `full + TGSD v2`
  - X shape: `130`
  - accuracy: `0.824`
  - precision: `1.0000`
  - recall: `0.450`
  - F1: `0.6207`

- `full + TGSD v2 + LFTP`
  - X shape: `642`
  - accuracy: `0.840`
  - precision: `0.9545`
  - recall: `0.525`
  - F1: `0.6774`

### Lettura test completi

- `TGSD v2` aveva gia migliorato molto il baseline
- `LFTP` ha migliorato ancora il recall sui live runtime
- il miglior setup completo resta `full + TGSD v2 + LFTP`

### Tempi test completi `full + TGSD v2 + LFTP`

- training dataset loading totale: `32331.87s`
- classifier training: `676.89s`
- training totale: `33008.77s`
- evaluation totale: `68.39s`
- esperimento totale: `33077.16s`

## Test fast

### Setup di riferimento storico

- train mode: `combined_runtime`
- eval mode: `runtime`
- experiment speed: `fast`
- CASIA sample ratio: `0.10`
- Replay sample ratio: `0.10`
- CASIA pre-extraction balance: `True`
- Replay pre-extraction balance: `True`
- runtime repeat: `1`

### Confronto fast storico vs completo sul nuovo `full + TGSD v2 + LFTP`

- `full`
  - train samples: `59761`
  - accuracy: `0.840`
  - precision: `0.9545`
  - recall: `0.525`
  - F1: `0.6774`

- `fast`
  - train samples: `4212`
  - accuracy: `0.792`
  - precision: `0.9375`
  - recall: `0.375`
  - F1: `0.5357`

### Lettura test fast storici

- il setup `fast` serve per iterare molto piu rapidamente del benchmark completo
- pero tende a perdere soprattutto sui live runtime
- la nuova campagna A/B ha mostrato che, nel fast mode, tenere CASIA non bilanciato e meglio del fast storico bilanciato

## Supporto implementazione PAD

### Stato fissato

| Blocco | Stato | Note | Priorita |
| --- | --- | --- | --- |
| **Freeze sperimentale** | **Quasi si** | Train/eval mode, split, metriche e pipeline sono gia abbastanza stabili. Manca pero una formalizzazione esplicita della baseline e della matrice esperimenti. | **Massima** |
| **Baseline** | **Si** | **LBP + quality + LFT-summary**, multiscala. Dimensione totale attuale: **128**. Questo e il riferimento fisso da non toccare finche non hai confronti puliti. | **Massima** |
| **TGSD_style / TGSD_light** | **Si, validato come supporto** | Struttura coerente: `I`, `blur(I)`, gradient similarity, top `%`, std finale. Output attuale: **+2** feature totali, una per scala. Validazione iniziale fatta su campione runtime: segnale presente ma separazione debole, quindi va tenuto come feature di supporto e non come blocco risolutivo. | **Massima** |
| **LFT map** | **Si** | Base corretta del ramo statistico. Nel paper e il punto di partenza sia per GGD sia per LFTP. | **Massima** |
| **GGD(LFT)_light** | **No** | Primo blocco statistico utile. Output atteso: **+4** feature totali, due per scala. Va testato **prima come sostituzione concettuale** del blocco LFT-summary, non come accumulo casuale. Nel paper GGD viene applicato alla distribuzione della LFT map per ricavare 2 parametri per scala. | **Alta** |
| **LFTP directional_light** | **No** | Step eventuale successivo. Ha senso solo se il ramo statistico mostra segnale. Versione light sensata: poche direzioni, statistiche semplici, niente AGGD subito. | **Media** |
| **LFTP attuale FFT-based** | **Si, ma fuori filone** | Utile come debug/esperimento. Non e il LFTP del paper, quindi non e una base teorica valida per AGGD. | **Bassa** |
| **AGGD(LFTP)** | **No** | Da rimandare. Nel paper AGGD e applicato agli istogrammi delle LFTP maps direzionali; senza una LFTP strutturalmente corretta, qui stai costruendo sul nulla. | **Bassa per ora** |
| **Matrice esperimenti formalizzata** | **No** | Essenziale per confronti puliti. Va definita nel runner, non improvvisata ogni volta. | **Alta** |
| **Report comparativi standardizzati** | **No** | Necessari per decidere con criterio: recall live, confusion matrix, precision, F1, accuracy, tempi. | **Alta** |

### Regole fisse

- `Baseline = 128`
- `TGSD_light = +2`
- `GGD(LFT)_light = +4`
- `PAD_DEBUG_FEATURE_MODE = baseline | full`
- `PAD_LFT_MODE = summary | ggd_light`
- `GGD(LFT)_light` va testato prima come sostituzione concettuale del blocco `LFT-summary`
- `baseline con GGD(LFT) al posto del summary LFT` significa:
  `LBP + quality + GGD(LFT)` multiscala
- non significa:
  `LBP + quality + LFT-summary + GGD(LFT)`
- `AGGD(LFTP)` non si tocca finche non esiste una `LFTP` corretta
- `TGSD_light` va considerato `TGSD-style` finche non supera validazione di segnale e stabilita

### Roadmap operativa

Stato corrente:
- `Blocco 1 - Validazione TGSD` completato
- prossimo step attivo: `Blocco 2 - GGD(LFT)_light`

#### Blocco 0 - Freeze sperimentale

Obiettivo:
- bloccare tutto quello che non sono le feature

Cosa fissare:
- dataset e split
- pipeline offline = runtime
- preprocessing
- SVM + scaler
- metriche ufficiali

Metriche ufficiali:
1. Recall live
2. Confusion matrix
3. Precision
4. F1
5. Accuracy

#### Blocco 1 - Validazione TGSD

Stato:
- gia implementato quasi del tutto

Da verificare:
- segnale: media TGSD live vs spoof
- stabilita: stesso volto -> valori simili, piccoli cambi -> piccoli cambi

Esito:
- validazione iniziale positiva ma debole
- `TGSD_light` resta nel ramo principale con `+2` feature
- non e il blocco da ottimizzare ora

#### Blocco 2 - GGD(LFT)_light

Obiettivo:
- introdurre modellazione statistica vera con costo basso

Setup concettuale:
- partire dalla `LFT map`
- stimare per ogni scala `shape` e `variance`
- ottenere `+4` feature totali
- usare:
  `shape = mean(|x|) / (sqrt(mean(x^2)) + eps)`
  `variance = var(x)`

Regola critica:
- primo test come sostituzione concettuale del blocco `LFT-summary`

#### Blocco 3 - Matrice esperimenti fissa

Esperimenti consentiti nel primo giro:
1. `Baseline` -> `128`
2. `Baseline + TGSD` -> `130`
3. `Baseline senza LFT-summary + GGD(LFT)` -> circa `124`
4. `Baseline + TGSD + GGD(LFT)` -> circa `126`

Nota:
- le dimensioni `124` e `126` assumono che `GGD(LFT)` sostituisca gli `8` valori totali di `LFT-summary` con `4` valori statistici
- quindi:
  - `baseline + PAD_LFT_MODE=summary` -> `128`
  - `baseline + PAD_LFT_MODE=ggd_light` -> `124`

#### Blocco 4 - Report standard

Per ogni esperimento salvare sempre:
- recall live
- precision spoof
- confusion matrix
- F1
- accuracy
- tempo esecuzione

#### Blocco 5 - Decisione

Se `GGD(LFT)` migliora il recall live:
- allora ha senso estendere il ramo statistico verso `LFTP directional_light`

Se `GGD(LFT)` non cambia nulla:
- fermarsi e rivalutare
- non ha senso saltare direttamente ad `AGGD` per fede

#### Blocco 6 - Solo dopo

Ordine eventuale:
1. `LFTP directional_light`
2. validazione delle distribuzioni
3. `AGGD(LFTP)`

### Sintesi operativa

- pipeline: OK
- spoof: OK
- live borderline: problema aperto
- priorita reale: feature con segnale reale, non accumulo di feature
- ordine reale:
  1. freeze
  2. validare TGSD
  3. implementare GGD(LFT)
  4. fare 4 test controllati
  5. decidere se continuare sul ramo `LFTP / AGGD`
