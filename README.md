# Simulatore di Guida Autonoma 2D con PyTorch (DQN)

Questo progetto implementa una simulazione 2D in Python in cui una macchinina deve imparare a percorrere un circuito nel minor tempo possibile senza uscire di strada. L'agente viene addestrato tramite Reinforcement Learning (in particolare, l'algoritmo **Deep Q-Network - DQN**) usando **PyTorch**, **Gymnasium** e **Pygame**.

## Struttura del Progetto

Il progetto contiene due filoni di addestramento: **Deep Q-Learning (DQN)** e **Algoritmo Genetico (Neuroevoluzione)**.

### File Comuni e DQN:
- **`environment.py`**: Definisce il circuito (curve spline Catmull-Rom) e l'ambiente Gymnasium (`CarRacingEnv`) per un singolo veicolo. Gestisce la fisica, il raycasting e il disegno.
- **`model.py`**: Definisce la rete neurale PyTorch utilizzata sia da DQN che dall'Algoritmo Genetico.
- **`agent.py`**: Gestisce l'agente DQN (scelta azioni, replay buffer, aggiornamenti e caricamento/salvataggio dei modelli).
- **`train.py`**: Script per addestrare l'agente via DQN (in modalità headless). Salva il modello in `models/best_model.pt`.
- **`enjoy.py`**: Visualizza la macchinina in azione con la grafica. Supporta la guida manuale o l'AI (sia DQN che Genetico).

### File Algoritmo Genetico:
- **`genetic_environment.py`**: Estende la simulazione per gestire una popolazione parallela di vetture contemporaneamente.
- **`train_genetic.py`**: Script per addestrare i pesi della rete neurale tramite selezione naturale, crossover e mutazione di una popolazione di auto.

---

## Installazione ed Avvio

1. **Crea l'ambiente virtuale** (già fatto se usi la configurazione corrente):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Come Usare il Progetto

### 1. Addestrare il Modello
Per avviare l'addestramento della rete neurale, esegui:
```bash
python train.py
```
*Nota: Questa modalità è "headless" (senza finestra grafica) per addestrare l'agente molto rapidamente. Durante l'esecuzione vedrai stampate le metriche sulle performance correnti ogni 10 episodi. Al termine, troverai il grafico dei progressi in `models/training_progress.png`.*

### Continuare l'Addestramento di un Modello Esistente
Se desideri continuare l'addestramento partendo da un modello già salvato (es. `models/final_model.pt` o uno dei checkpoint salvati), puoi passare il percorso del file come parametro dello script. L'agente caricherà i pesi della rete, lo stato dell'ottimizzatore e il contatore degli step riprendendo l'addestramento con lo stesso livello di esplorazione ($\epsilon$) del momento del salvataggio:
```bash
python train.py models/final_model.pt
```

---

### Addestrare con l'Algoritmo Genetico (Neuroevoluzione)
In alternativa al DQN, l'Algoritmo Genetico fa correre una popolazione di auto (es. 40) contemporaneamente. A ogni generazione, le auto peggiori (che sbattono o si fermano) vengono eliminate. Le auto migliori (con maggiore "fitness") vengono selezionate per riprodursi tramite crossover e mutazione dei pesi delle loro reti neurali.

Questo approccio è eccezionalmente stabile per aggirare curve complesse perché la popolazione esplora molteplici traiettorie in parallelo.

#### Addestramento Headless (molto veloce):
```bash
python train_genetic.py
```

#### Addestramento Visivo (consigliato per lo spettacolo!):
Puoi guardare l'intera popolazione di macchine che corre e impara in tempo reale sullo schermo:
```bash
python train_genetic.py --render
```
*Le auto che sbattono diventano rosse e trasparenti, mentre il leader della generazione corrente viene evidenziato in azzurro proiettando i suoi sensori laser.*

Al termine dell'addestramento, il modello migliore viene salvato in `models/genetic_best.pt` e il grafico dei progressi in `models/genetic_progress.png`. Per preservare i checkpoint storici, il sistema salva anche file come `models/checkpoint_gen_20.pt` ogni 20 generazioni.

#### Continuare il Training Genetico da un Modello Preesistente:
Se desideri riprendere l'addestramento o affinare un modello genetico (o persino un modello addestrato in DQN!) usando la neuroevoluzione, puoi passare il percorso del file come argomento:
```bash
python train_genetic.py models/genetic_best.pt
# oppure con rendering attivo:
python train_genetic.py models/genetic_best.pt --render
```
*L'algoritmo caricherà il modello e lo userà per "seminare" la prima generazione: una copia esatta del campione (per non perdere le abilità acquisite) e le restanti auto come cloni mutati del campione (con gradi di mutazione variabili per esplorare lo spazio circostante).*

Per guardare il modello genetico migliore all'opera:
```bash
python enjoy.py models/genetic_best.pt
```


---

### 2. Guardare il Modello Addestrato o Guidare Manualmente
Per visualizzare l'interfaccia grafica e vedere come guida il modello addestrato (oppure guidare tu stesso):
```bash
python enjoy.py
```

#### Controlli Tastiera:
- **`A`**: Passa alla modalità **Guida Autonoma (AI)**.
- **`M`**: Passa alla modalità **Guida Manuale**.
- **`R`**: Resetta la macchinina all'inizio del circuito.
- **Freccia Su / W**: Accelera.
- **Freccia Giù / S**: Frena / Retromarcia.
- **Freccia Sinistra / A**: Curva a sinistra.
- **Freccia Destra / D**: Curva a destra.
- **`ESC`**: Chiude il simulatore.

---

## Dettagli Tecnici

- **Sensori (Input dell'Agente)**: La macchina proietta 7 sensori di distanza (raggi) con angoli a `[-90°, -45°, -20°, 0°, 20°, 45°, 90°]` rispetto alla direzione del veicolo. Questi raggi rilevano la distanza dai bordi stradali. L'input finale dell'agente è composto da questi 7 valori normalizzati più la velocità corrente della vettura.
- **Spazio delle Azioni (Output)**: L'agente ha a disposizione 9 azioni discrete derivate dalla combinazione di sterzo (Sinistra, Dritto, Destra) e acceleratore (Accelerazione, Coasting, Freno).
- **Funzione di Reward**:
  - Penalità fissa a ogni step per incoraggiare la velocità e il completamento rapido.
  - Ricompensa per il superamento sequenziale dei checkpoint del circuito (`+15`).
  - Ricompensa per il completamento di ogni giro (`+100`).
  - Ricompensa per l'allineamento della direzione della macchina rispetto alla tangente del circuito.
  - Grande penalità in caso di collisione con i bordi del circuito (`-100`).
