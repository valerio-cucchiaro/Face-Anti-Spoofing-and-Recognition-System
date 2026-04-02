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
- augmentation scope promossa: `runtime_only`
- stato:
  - validata in `fast`
  - da verificare ancora in `full`

### Configurazione in uso adesso

- feature set: `baseline + TGSD + GGD(LFT)_light`
- TGSD: `tau170 | sobel | top5`
- augmentation training: `runtime-only mix`
- questa e` la configurazione corrente da considerare riferimento operativo

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
| `full` current setup | `baseline + TGSD + GGD(LFT)_light` + `runtime-only mix` | `TBD` | `TBD` | `TBD` | **da lanciare e compilare** |

Legenda metrica nelle celle:
- `accuracy / precision / recall / f1`

### Setup del prossimo full da eseguire

- train mode: `combined_runtime`
- eval mode: `all`
- experiment speed: `full`
- feature set: `baseline + TGSD + GGD(LFT)_light`
- `tgsd_tau = 170`
- `tgsd_gradient_operator = sobel`
- `tgsd_top_percent = 0.05`
- augmentation mode: `mix`
- augmentation scope: `runtime_only`

### Nota

- il `full` attualmente documentato conferma bene il feature set
- il vero `full` coerente con la configurazione corrente deve ancora essere eseguito con `runtime-only mix`

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
  - scope `runtime_only`
- miglior risultato `fast` attuale:
  - `0.880 / 1.0000 / 0.625 / 0.7692`
  - confusion matrix `[[85, 0], [15, 25]]`

## Roadmap Operativa

### Stato reale

- pipeline PAD: stabile
- feature work: chiuso
- spoof: forte
- live: ancora il problema aperto
- setup `fast` attuale: definito
- setup `full` con configurazione attuale completa: ancora da verificare

### Prossimi passi

1. eseguire il `full` con:
   - `baseline + TGSD + GGD(LFT)_light`
   - `tau170 | sobel | top5`
   - `runtime-only mix`
2. aggiornare la tabella `full` con i risultati reali
3. dopo il `full`, decidere se il problema residuo richiede:
   - analisi piu` profonda dei `live` falsi negativi
   - oppure lavoro mirato sui dati runtime/live

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
  - `runtime-only mix`
- stato validazione:
  - `fast`: validato
  - `full`: da completare con la stessa configurazione
- problema aperto reale:
  - robustezza dei `live`
