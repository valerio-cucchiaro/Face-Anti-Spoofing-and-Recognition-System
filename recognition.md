# Recognition

## Setup attuale

- trainer: `src/recognition/recognition_model_trainer.py`
- evaluator: `src/recognition/recognition_model_evaluator.py`
- runner: `src/recognition/recognition_experiment_runner.py`
- runtime smoothing: `src/recognition/recognition_runtime_state.py`
- pipeline runtime di riferimento per la sola recognition: `src/pipeline/run_pipeline_bs.py`
- pipeline integrata PAD + recognition: `src/pipeline/run_pipeline_full.py`

### Configurazione recognition attuale

- modello: `LBPH`
- parametri LBPH di riferimento:
  - `radius=1`
  - `neighbors=8`
  - `grid_x=6`
  - `grid_y=6`
- preprocessing di riferimento:
  - normalizzazione: `equalize`
  - crop padding ratio: `0.10`
- artefatti canonici:
  - `models/lbph_model.xml`
  - `models/label_map.pkl`
  - `models/recognition_config.json`

### Threshold attuali

- soglia runtime webcam di riferimento in `full`:
  - `44.0`
- `bs` e` il banco prove della recognition:
  - usa soglie sperimentali nei test live
  - i run sotto servono a confrontare il comportamento al variare della soglia
- soglia offline salvata in `models/recognition_config.json`:
  - `29.4611`
  - source: `custom_orl_to_orl`

Nota:

- la soglia offline e la soglia runtime non coincidono volutamente
- la prima e` una reference evaluator-side
- la seconda e` la soglia live stabilizzata nei test webcam

### Dataset supportati

- `custom`
- `orl`
- `lfw`
- `custom_orl`
- `all`

## Test di riferimento

| Run | Gallery / identities | Probe | Accuracy | Genuine | Impostor | Threshold suggerita | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `custom -> custom` | `60 / 2` | `20` | `1.0000` | `20` | `0` | `104.4171` fallback | sanity check custom |
| `custom_orl -> custom` | `260 / 42` | `20` | `1.0000` | `20` | `0` | `45.4040` fallback | mix realistico, perfetto su custom |
| `orl -> orl` | `200 / 40` | `200` | `0.9200` | `184` | `16` | `89.9169` midpoint | run valido per threshold offline |
| `custom_orl -> orl` | `260 / 42` | `200` | `0.9200` | `184` | `16` | `89.9169` midpoint | run valido per threshold offline |
| `all -> custom` | `3788 / 943` | `20` | `0.8500` | `17` | `3` | `88.9671` midpoint | confronto completo con LFW |

## Test extra e storico

### Ultimo run utile su custom aggiornato

| Run | Gallery / identities | Probe | Accuracy | Genuine | Impostor | Threshold suggerita | Tempo totale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `custom_orl -> custom` | `260 / 42` | `20` | `1.0000` | `20` | `0` | `45.4040` | `1.35s` |

### Test runtime BS (prove)

| Scenario | Threshold | Lettura sintetica | Nota |
| --- | --- | --- | --- |
| `valerio` senza luce laterale | `44.0` | `accepted=67`, `rejected=0`, `accepted_confidence=mean=40.52`, `min=38.95`, `max=42.28` | molto stabile, known sempre confermato |
| `valerio` senza luce laterale | `40.0` | `accepted=72`, `rejected=84`, `accepted_confidence=mean=38.86`, `rejected_confidence=mean=40.84` | troppo severa, taglia parte del known |
| `valerio` senza luce laterale | `41.0` | `accepted=94`, `rejected=16`, `accepted_confidence=mean=39.10`, `rejected_confidence=mean=42.15` | compromesso migliore, match pulito |
| `valerio` con forte luce laterale | `44.0` | `accepted=4`, `rejected=241`, `rejected_confidence=mean=46.96` | condizione molto più instabile, tende a `Unknown` |

### Confronto CLAHE runtime

| Scenario | Threshold | Lettura sintetica | Nota |
| --- | --- | --- | --- |
| `valerio` con luce media, CLAHE attivo | `43.0` | `accepted=0`, `rejected=61`, `rejected_confidence=mean=45.83` | CLAHE runtime peggiora la recognition, da scartare |
| `valerio` con luce media, CLAHE disattivo | `43.0` | `accepted=53`, `rejected=17`, `accepted_confidence=mean=41.83`, `rejected_confidence=mean=43.95` | base senza CLAHE molto piu` sana |

### Confronto gamma runtime

| Scenario | Threshold | Lettura sintetica | Nota |
| --- | --- | --- | --- |
| `valerio` con luce media, gamma runtime attiva | `43.0` | `accepted=0`, `rejected=33`, `rejected_confidence=mean=45.59`, `light_gamma_outcome=better=8`, `worse=21`, `equal=4` | la correzione gamma non porta un guadagno medio, da scartare |
| `valerio` con luce media, base senza correzioni runtime | `43.0` | `accepted=53`, `rejected=17`, `accepted_confidence=mean=41.83`, `rejected_confidence=mean=43.95` | base migliore della gamma runtime |

### Run storico `all -> all`

| Eval subset | Probe | Accuracy | Genuine | Impostor | Threshold suggerita | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `custom` | `10` | `0.7000` | `7` | `3` | `94.7661` | custom iniziale, prima dell'espansione |
| `orl` | `200` | `0.9200` | `184` | `16` | `104.5012` | ORL stabile |
| `lfw` | `4078` | `0.0692` | `282` | `3796` | `80.4652` | stress test, limite strutturale di LBPH |

### Tuning LBPH riassunto

| Configurazione | `custom_orl -> custom` | `custom_orl -> orl` | Lettura |
| --- | --- | --- | --- |
| `r1 n8 gx6 gy6` | `1.0000` | `0.9600`, `accepted_impostor=4`, `rejected_genuine=5` | migliore configurazione operativa |
| `r2 n8 gx6 gy6` | `1.0000` | `0.9700`, `accepted_impostor=4`, `rejected_genuine=11` | buona ma piu` severa sui genuine |
| `r1 n8 gx8 gy8` | `1.0000` | `0.9200`, `accepted_impostor=8`, `rejected_genuine=7` | peggiore su ORL |
| `r2 n8 gx8 gy8` | `1.0000` | `0.9400`, `accepted_impostor=4`, `rejected_genuine=8` | non batte `gx6 gy6` |
| `r1 n8 gx10 gy10` | `1.0000` | `0.9200`, `accepted_impostor=8`, `rejected_genuine=12` | troppo rumorosa |
| `r2 n8 gx10 gy10` | `0.9500` | `0.9000`, `accepted_impostor=7`, `rejected_genuine=13` | bocciata |

### Tuning preprocessing riassunto

| Test | Config | `custom_orl -> custom` | `custom_orl -> orl` | Esito |
| --- | --- | --- | --- | --- |
| normalizzazione | `equalize` | `0.95` | `0.97` | scelta finale |
| normalizzazione | `CLAHE` | `0.95` | `0.95` | provato sul runtime, peggiora e non conviene |
| normalizzazione | `none` | `1.00` | `0.965` | buono su custom, meno difendibile su ORL |
| padding | `0.00` | `0.95` | `0.97` | buono ma inferiore |
| padding | `0.05` | `0.90` | `0.96` | bocciato |
| padding | `0.10` | `1.00` | `0.975` | scelta finale |
| padding | `0.15` | `1.00` | `0.935` | peggiora su ORL |

## Decisioni correnti

- recognition attiva: `LBPH`
- configurazione LBPH promossa:
  - `radius=1`
  - `neighbors=8`
  - `grid_x=6`
  - `grid_y=6`
- preprocessing promosso:
  - `equalize`
  - `padding ratio = 0.10`
- training misto di riferimento:
  - `custom_orl`
- run evaluator-side di riferimento per threshold offline:
  - `orl -> orl`
  - `custom_orl -> orl`
- `LFW` resta solo stress test
- runtime webcam di riferimento:
  - `bs` per testare la recognition da sola
  - `full` per il comportamento integrato con PAD

## Stato corrente

### Cose chiuse

- pipeline recognition end-to-end
- configurazione LBPH di riferimento
- preprocessing di riferimento
- smoothing temporale runtime
- separazione tra threshold offline e threshold live
- runtime `bs` reso visivamente piu` stabile

### Cose stabili

- `custom_orl` come training misto realistico
- `orl -> orl` e `custom_orl -> orl` come run piu` difendibili per la soglia offline
- `bs` come ambiente giusto per osservare la recognition senza PAD

### Cose ancora aperte

- stabilita` del crop del volto e dell'estrazione dei dettagli prima della recognition
- confidence piu` robusta oltre `threshold + voti temporali`
- logging runtime piu` strutturato nella pipeline `full`
- valutazione open-set piu` esplicita

## Roadmap operativa

1. migliorare la stabilita` del crop e dell'estrazione del volto prima della recognition
2. raccogliere e ricalibrare la threshold runtime sui valori reali del runtime live
3. ripulire la gallery e verificare la qualita` delle immagini per identita`
4. controllare coerenza tra `run_pipeline_bs.py` e `run_pipeline_full.py` lato recognition
5. ripulire eventuali residui di vecchie configurazioni o test nel ramo recognition
6. verificare file/funzioni legacy non piu` usati

## Roadmap runtime-only

Questa sotto-roadmap raccoglie i passi che migliorano la recognition in
runtime senza toccare il preprocessing salvato nel dataset e senza
richiedere una rigenerazione dei sample recognition.

1. usare solo il modello `custom` come baseline runtime di confronto
2. fare uno sweep delle soglie runtime live e confrontare i risultati
3. introdurre un filtro piu` severo sui frame borderline prima della prediction
4. alzare la severita` del filtro temporale prima del `final_name`
5. rafforzare la logica di lock / isteresi per il nome mostrato in webcam
6. aggiungere log runtime piu` strutturati su decisione, candidato, final e stabilita`
7. confrontare i log di `bs` e `full` per verificare che il comportamento sia coerente
8. solo dopo, decidere se il problema residuo e` davvero di preprocessing o di dataset

## Nota metodologica

- l'obiettivo non e` rendere `LBPH` competitivo su `LFW`
- l'obiettivo e` mantenere una recognition coerente, difendibile e stabile sul caso reale
- `custom -> custom` resta sanity check
- `custom_orl -> orl` resta il run piu` utile per leggere la separazione offline
- `all -> custom` resta confronto completo, non riferimento operativo
