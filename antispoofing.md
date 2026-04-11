# Confronto PAD Runtime

## Setup attuale

### Feature set promosso

- feature set: `baseline + TGSD + GGD(LFT)_light`
- `PAD_DEBUG_FEATURE_MODE = baseline | tgsd`
- `PAD_LFT_MODE = ggd_light`
- `tgsd_tau = 170`
- `tgsd_gradient_operator = sobel`
- `tgsd_top_percent = 0.05`

### Training augmentation promossa

- augmentation live fotometrica: `mix`
- augmentation scope promossa in `fast`: `runtime_only`
- stato:
  - validata in `fast`
  - `full` migliore ora: `mix` su tutti i live

### Configurazione in uso adesso

- feature set: `baseline + TGSD + GGD(LFT)_light`
- TGSD: `tau170 | sobel | top5`
- augmentation training di riferimento `fast`: `runtime-only mix`
- PAD runtime: finestra `5` piena prima della decisione
- PAD runtime: `enter_live = 0.65`, `enter_spoof = 0.30`
- PAD runtime: `PAD_RUN_INTERVAL = 2` in `fast` e `full`
- prossimo test `full` da provare: `mix` su tutti i live

## Test Fast

### Tabella riassuntiva

| Run | Configurazione | Accuracy | Precision | Recall | F1 | Confusion Matrix | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fast_A_r1` | storico A/B, pre-augmentation | `0.832` | `0.9524` | `0.500` | `0.6557` | `[[84, 1], [20, 20]]` | miglior A/B iniziale |
| `baseline + GGD(LFT)_light` | `baseline` + `ggd_light` | `0.792` | `0.8889` | `0.400` | `0.5517` | `[[83, 2], [24, 16]]` | non promosso |
| `baseline + TGSD` | `tgsd` + `summary` storico | `0.832` | `0.9524` | `0.500` | `0.6557` | `[[84, 1], [20, 20]]` | storico, ora archiviato |
| `baseline + TGSD + GGD(LFT)_light` | feature set leggero promosso | `0.848` | `0.9565` | `0.550` | `0.6984` | `[[84, 1], [18, 22]]` | miglior feature-only |
| `baseline + GGD(LFT)_true` | `baseline` + `ggd_true` | `0.792` | `0.8889` | `0.400` | `0.5517` | `[[83, 2], [24, 16]]` | archiviato |
| `baseline + TGSD + GGD(LFT)_true` | `tgsd` + `ggd_true` | `0.840` | `0.9545` | `0.525` | `0.6774` | `[[84, 1], [19, 21]]` | archiviato |
| `mix` | augmentation su tutti i live di training | `0.872` | `0.9615` | `0.625` | `0.7576` | `[[84, 1], [15, 25]]` | miglior augmentation generale |
| `gamma` | augmentation singola | `0.856` | `0.9583` | `0.575` | `0.7188` | `[[84, 1], [17, 23]]` | aiuta un po' |
| `brightness_contrast` | augmentation singola | `0.848` | `0.9565` | `0.550` | `0.6984` | `[[84, 1], [18, 22]]` | neutra |
| `side_shading` | augmentation singola | `0.856` | `0.9231` | `0.600` | `0.7273` | `[[83, 2], [16, 24]]` | interessante, ma apre 1 spoof in piu` |
| `mix + light_asymmetry` | `mix` + 2 feature luce | `0.856` | `0.9231` | `0.600` | `0.7273` | `[[83, 2], [16, 24]]` | bocciato |
| `runtime-only mix` | `mix`, solo live runtime | `0.880` | `1.0000` | `0.625` | `0.7692` | `[[85, 0], [15, 25]]` | **riferimento fast attuale** |
| `runtime-only mix + runtime_repeat=2` | `runtime_only` + repeat maggiore | `0.856` | `0.9231` | `0.600` | `0.7273` | `[[83, 2], [16, 24]]` | bocciato |

### Lettura

- il miglior `fast` attuale e`:
  - `baseline + TGSD + GGD(LFT)_light`
  - con `runtime-only mix`
- rispetto al solo feature set promosso senza augmentation:
  - recall live sale da `0.550` a `0.625`
  - F1 sale da `0.6984` a `0.7692`
  - gli spoof passano da `84/85` a `85/85`
- `runtime_repeat=2` non va promosso
- `light_asymmetry` non va promossa

## Test Full

### Tabella riassuntiva

| Run | Configurazione | CASIA | Replay | Runtime | Note |
| --- | --- | --- | --- | --- | --- |
| `full` feature validation | `baseline + TGSD + GGD(LFT)_light` con `tau170 | sobel | top5` | `0.9453 / 0.7699 / 0.7898 / 0.7797` | `0.9541 / 0.9462 / 0.8347 / 0.8870` | `0.8400 / 1.0000 / 0.5000 / 0.6667` | conferma del feature set promosso |
| `full` runtime-only mix | `baseline + TGSD + GGD(LFT)_light` + `runtime-only mix` | `0.9451 / 0.7693 / 0.7894 / 0.7792` | `0.9542 / 0.9466 / 0.8347 / 0.8871` | `0.8240 / 1.0000 / 0.4500 / 0.6207` | runtime-only non confermata |
| `full` all-live mix | `baseline + TGSD + GGD(LFT)_light` + `mix` su tutti i live | `0.9451 / 0.7693 / 0.7894 / 0.7792` | `0.9577 / 0.9319 / 0.8674 / 0.8985` | `0.8560 / 1.0000 / 0.5500 / 0.7097` | riferimento precedente |
| `full` all-live mix + nuovi live | `baseline + TGSD + GGD(LFT)_light` + `mix` su tutti i live | `0.9447 / 0.7538 / 0.8154 / 0.7834` | `0.9601 / 0.9363 / 0.8744 / 0.9043` | `0.8960 / 0.9355 / 0.7250 / 0.8169` | **miglior full attuale** |

Legenda metrica nelle celle:
- `accuracy / precision / recall / f1`

### Setup del full di verifica

- train mode: `combined_runtime`
- eval mode: `all`
- experiment speed: `full`
- feature set: `baseline + TGSD + GGD(LFT)_light`
- `tgsd_tau = 170`
- `tgsd_gradient_operator = sobel`
- `tgsd_top_percent = 0.05`
- augmentation mode: `mix`
- augmentation scope: `all`

### Nota

- il `full` attualmente documentato conferma bene il feature set
- `runtime-only mix` migliora il `fast` ma non e` stata confermata nel `full`
- il `full` migliore attuale e` `mix` su tutti i live, con i nuovi live aggiunti
- il runtime `full` ora usa lock piu` leggibile: `face_missing=2`, `movement=1`, `crop=1`
- `cv` e` allineato al `full` su PAD interval e soglia `enter_live`, ma mantiene i log diagnostici su distanza e luce

## Test Archiviati

### A/B iniziale `fast`

| Run | Accuracy | Precision | Recall | F1 | Confusion Matrix | Esito |
| --- | --- | --- | --- | --- | --- | --- |
| `fast_A_r1` | `0.832` | `0.9524` | `0.500` | `0.6557` | `[[84, 1], [20, 20]]` | migliore |
| `fast_B_r1` | `0.792` | `0.9375` | `0.375` | `0.5357` | `[[84, 1], [25, 15]]` | peggiore |
| `fast_A_r2` | `0.824` | `0.9500` | `0.475` | `0.6333` | `[[84, 1], [21, 19]]` | non migliora |
| `fast_A_r5` | `0.824` | `0.9500` | `0.475` | `0.6333` | `[[84, 1], [21, 19]]` | non migliora |

### Refinement TGSD archiviato

| Variante | Accuracy | Precision | Recall | F1 | Esito |
| --- | --- | --- | --- | --- | --- |
| `tau170 | sobel | top5 | baseline + TGSD` | `0.832` | `0.9524` | `0.500` | `0.6557` | riferimento TGSD puro |
| `tau170 | sobel | top5 | baseline + TGSD + GGD_light` | `0.848` | `0.9565` | `0.550` | `0.6984` | promosso |
| `tau170 | prewitt | top5 | baseline + TGSD` | `0.832` | `0.9524` | `0.500` | `0.6557` | nessun vantaggio |
| `tau170 | prewitt | top5 | baseline + TGSD + GGD_light` | `0.832` | `0.9130` | `0.525` | `0.6667` | peggiore |
| `tau170 | prewitt | top10 | baseline + TGSD` | `0.808` | `1.0000` | `0.400` | `0.5714` | peggiore |
| `tau170 | prewitt | top10 | baseline + TGSD + GGD_light` | `0.824` | `1.0000` | `0.450` | `0.6207` | peggiore |

### Feature e augmentation bocciate

| Test | Risultato | Esito |
| --- | --- | --- |
| `GGD(LFT)_true` | sotto o uguale a `GGD(LFT)_light` | archiviato |
| `LFT summary` | superato da `GGD(LFT)_light` | archiviato |
| `full mode` feature con `LFTP` | storico utile, ma non piu` nel ramo attivo | archiviato |
| `light_asymmetry` | `0.856 / 0.9231 / 0.600 / 0.7273` | bocciato |
| `brightness_contrast` singola | nessun guadagno pratico | bocciata |
| `runtime_repeat=2` con `runtime-only mix` | peggiora live e spoof | bocciato |

## Decisioni correnti

### Feature

- `final choice`:
  - `baseline + TGSD + GGD(LFT)_light`
  - `tau170 | sobel | top5`
- `feature work closed`
- rami archiviati:
  - `LFT summary`
  - `GGD(LFT)_true`
  - `full mode`
  - `Prewitt`
  - `top10`

### Augmentation

- augmentation promossa:
  - `mix`
  - scope `runtime_only` in `fast`
- miglior risultato `fast` attuale:
  - `0.880 / 1.0000 / 0.625 / 0.7692`
  - confusion matrix `[[85, 0], [15, 25]]`
- migliore `full` attuale:
  - `0.8560 / 1.0000 / 0.5500 / 0.7097`
  - confusion matrix `[[85, 0], [18, 22]]`

## Roadmap Operativa

### Stato reale

- pipeline PAD: stabile
- feature work: chiuso
- spoof: forte
- live: ancora il problema aperto
- setup `fast` attuale: definito
- `full` migliore attuale: `mix` su tutti i live
- runtime `full`: pulito dai log di debug, resta solo il flusso di fase
- runtime `cv`: usato come banco diagnostico per distanza, controluce e reattivita` del PAD

### Prossimi passi

1. tenere il `full` attuale come baseline operativa.
2. usare `cv` per verificare se il limite residuo e` davvero controluce/distanza o solo soglia.
3. raccogliere altri live mirati solo se i casi difficili restano riproducibili.
4. pulire i log debug quando il comportamento del runtime e` definitivamente confermato.

### Cose da non riaprire adesso

- nuove micro-feature sulla luce
- `GGD(LFT)_true`
- `LFT summary`
- `Prewitt`
- `top10`
- `full mode` feature con `LFTP`

## Sintesi

- configurazione attuale in uso:
  - `baseline + TGSD + GGD(LFT)_light`
  - `tau170 | sobel | top5`
  - `runtime-only mix` come riferimento `fast`
- stato validazione:
  - `fast`: validato con `runtime-only mix`
  - `full`: migliore `mix` su tutti i live, con i nuovi live aggiunti
- problema aperto reale:
  - robustezza dei `live`
  - controluce e distanza troppo corta

### Runtime PAD

- PAD window: `5`
- decisione solo a finestra piena
- soglie attuali:
  - `enter_live = 0.65`
  - `exit_live = 0.55`
  - `enter_spoof = 0.30`
  - `exit_spoof = 0.45`
- lock runtime:
  - `face_missing = 2`
  - `movement = 1`
  - `crop = 1`
- lettura pratica:
  - `live` entra piu` facilmente del vecchio `0.70`
  - `spoof` resta severo
  - controluce e foto da telefono sono gestiti bene, ma la webcam puo` ancora restare conservativa nei casi sporchi

## Osservazioni Runtime Live

| Scenario | Esito | Nota |
| --- | --- | --- |
| `no light` | migliore | quasi sempre `live` |
| `luce laterale` | borderline | ogni tanto fatica e oscilla |
| `luce dietro` | peggiore | va quasi sempre in `spoof` |
| `vicino alla webcam` | fragile | la distanza pesa molto |
| `tablet/foto` | corretto | spoof forte, recupero lento ma coerente |

- lettura attuale: il PAD e` robusto nel caso pulito, ma diventa conservativo con controluce e distanza ridotta
- prossima verifica: capire se il margine residuo su live va ancora corretto con nuovi dati o e` ormai accettabile per la demo
