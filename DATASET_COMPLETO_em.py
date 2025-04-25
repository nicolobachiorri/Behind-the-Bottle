import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import math
import matplotlib.pyplot as plt
import os
from PIL import Image

# SISTEMAZIONE DATASET
data_r = pd.read_csv(os.path.join("dataset", "winequality-red.csv"), sep=";")
data_w = pd.read_csv(os.path.join("dataset", "winequality-white.csv"), sep=";")

data_r["color"] = 'Red wine'
data_w["color"] = 'White wine'

data_tot = pd.concat([data_r, data_w], ignore_index=True)

# INTERFACCIA
st.title('Wine Quality Data Visualization')

# Creazione delle schede
tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["About the Dataset", "Feature", "Feature & Target", "Analisi binaria", "Analisi a 3 categorie", "Analisi colore"])

# Separazione di covariate da variabili
covariate = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
          'pH', 'sulphates', 'alcohol']
columns = data_tot.columns.tolist()

# Dizionario con le descrizioni
descrizioni = {
    "fixed acidity": "La distribuzione dei vini bianchi risulta avere una distribuzione più concentrata, e la forma ricorda una gaussiana.\n Il vino rosso invece ha una distribuzione più ampia, che suggerisce maggiore variabilità. Inoltre presenta un’asimmetria verso destra.",
    "volatile acidity": "Differenze significative fra le due distribuzioni. La distribuzione dei vini bianchi tende and una gaussiana, con la coda destra leggermente più allungata; in media un vino bianco ha un’acidità volatile di circa 0.3 g/L. La distribuzione dei vini rossi, oltre ad avere una variabilità più elevata ha in media un’acidità volatile superiore, intorno a 0.5 g/L.",
    "citric acid": " Differenze significative fra le due distribuzioni. La distribuzione dell’acido citrico nei vini rossi sembra essere quasi costante, mentre la distribuzione dei vini bianchi è più tendente ad una gaussiana.",
    "residual sugar": "Le distribuzioni sono molto simili, ma la distribuzione dei vini bianchi ha una coda più lunga rispetto a quella dei vini rosssi.",
    "chlorides": "Le distribuzioni dei vini ricordano entrambe una gaussiana ma quella dei vini rossi è leggermente spostata:  i vini rossi in media hanno un livello di cloruro più alto rispetto ai vini bianchi",
    "free sulfur dioxide": "La distribuzione dei vini rossi è concentrata intorno ai livelli di diossido solfurico più bassi,, mentre quella dei vini bianchi è più variabile, con un livello medio di diossido solfurico più.elevato.",
    "total sulfur dioxide": "Le distribuzioni hanno mode estremamente diverse: i vini rossi tendono ad avere livelli totali di diossido solfurico più bassi rispetto ai vini bianchi. Inoltre la distribuzione dei vini bianchi ha più variabilità.",
    "density": "La distribuzione dei vini rossi è più ristretta, mentre quella dei vini bianchi presenta maggiore variabilità. La moda per i vini bianchi è inferiore a quella dei vini rossi.",
    "pH": "Entrambe le distribuzioni hanno forme che ricordano una gaussiana. La distribuzione dei vini rossi è leggermente asimmetrica verso sinistra e presenta una moda pari a 3.3 circa. I vini bianchi invece hanno in media un livello di ph inferiore, intorno a 3,2.",
    "sulphates": "La distribuzione dei vini rossi è spostata verso valori più elevati rispetto al vino bianco e presenta una coda più lunga verso destra. La distribuzione dei vini bianchi risulta più simmetrica. Generalmente un vino bianco avrà una concentrazione di solfato più bassa rispetto ad un ",
    "alcohol": "Distribuzioni molto simili, i vini rossi tendono ad avere una quantità di alcol più alta mentre i vini bianchi si concentrano su valori inferiori."
    }

with tab1:
    st.title('About this Dataset')
    st.markdown("""
    ### Descrizione dei Dataset

    I dataset analizzati in questo lavoro sono 3: uno relativo ai vini bianchi, un altro contenente le istanze dei vini rossi e infine l’ultimo dataset è il risultato della concatenazione dei due (è stata aggiunta una nuova variabile per distinguere un vino bianco da uno rosso). L’insieme dei dati descrive le caratteristiche chimiche e sensoriali di vini attraverso una serie di variabili numeriche.

    #### Variabili del Dataset

    - *Fixed acidity*: acidità fissa
    - *Volatile acidity*: acidità volatile 
    - *Citric acid*: acido citrico
    - *Residual sugar*: zuccheri residui 
    - *Chlorides*: sodio cloruro.
    - *Free sulfur dioxide*: anidride solforosa libera 
    - *Total sulfur dioxide*: totale di SO₂ 
    - *Density*: densità del vino    
    - *PH*: misura dell’acidità generale.
    - *Sulphates*: solfati
    - *Alcohol*: percentuale di alcol nel vino (% vol).
    - *Quality*: valutazione sensoriale della qualità del vino (questa variabile è stata studiata sia in modo binario: if quality <=5 then quality = 0, if quality > 5 then quality = 1; sia suddiviso in tre classi: basso (punteggi da 1-4), medio (5-6) e alto (7-8-9)
    - *Colour*: tipo di vino (presente solo nel dataset concatenato)


    ---

    ### Progetto di Data Visualization

    *Obiettivo*: 
    - Creare rappresentazioni visive del dataset per comunicare informazioni in modo chiaro ed efficace.

    *Attività*:
    - Sviluppare grafici e visualizzazioni per evidenziare andamenti, distribuzioni e relazioni tra le variabili.

    *Deliverables*:
    - Una serie di visualizzazioni che mettono in luce gli aspetti più interessanti del dataset.
    - Una narrazione a supporto, che guida l'utente nell'interpretazione dei dati e ne spiega la rilevanza.
    """)
    
    st.markdown("---")
    numeric_df = data_tot.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    st.subheader('Matrice di Correlazione')
    st.markdown('Viene riportata la matrice di correlazione relativa al dataset contenente sia i vini bianchi che rossi. Due variabili sono considerate notevolmente correlate se presentano valori superiori a |0.6|.')
    # Percorso dell'immagine salvata
    corr_img_path = f"images/corr_matrix.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(corr_img_path):
        img = Image.open(corr_img_path)
        st.image(img, caption=f"Matrice di Correlazione")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, 
                annot=True, 
                ax=ax, 
                cmap="coolwarm", 
                fmt=".2f",
                alpha=0.6
                )
        fig.tight_layout()
        fig.savefig(corr_img_path)
        st.pyplot(fig)


    st.markdown("---")

with tab2:  
    st.title('Feature')

    col1, col2 = st.columns(2)
    var_4 = col1.selectbox("Select a variable", covariate, key="selectbox_var_4")
    st.subheader(f"Distribuzione di {var_4} per tipo di vino")
    # Percorso dell'immagine salvata
    img_path = f"images/hist_{var_4.replace(' ', '_')}.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Distribuzione di {var_4}")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=data_tot,
            x=var_4,
            hue="color",
            kde=True,
            bins=30,
            palette=['purple', 'lightgray'],
            alpha=0.6,
            ax=ax
        )
        plt.title(f"Distribuzione di {var_4} per tipo di vino")
        fig.tight_layout()
        fig.savefig(img_path)
        st.pyplot(fig)  

    st.markdown("**Confronto fra la distribuzione dei vini bianchi e la distribuzione dei vini rossi, si ricorda che la numerosità di osservazioni nel dataset dei vini bianchi è superiore a quella dei vini rossi:**")
    if var_4 in descrizioni:
        st.markdown(f"{descrizioni[var_4]}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    var_1 = col1.selectbox("Select the firs variable (x)", covariate, key="selectbox_var_1")
    var_2 = col2.selectbox("Select the second variable (y)", [col for col in covariate if col != var_1], key="selectbox_var_2")
    st.subheader(f"Scatter Plot: {var_1} vs {var_2}")
    st.markdown('Per osservare meglio le singole correlazioni fra coppie di variabili si propone uno scatterplot bivariato.')
    
    # Percorso dell'immagine salvata
    scatt_img_path = f"images/scatter_{var_1.replace(' ', '_')}_vs_{var_2.replace(' ', '_')}.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(scatt_img_path):
        img = Image.open(scatt_img_path)
        st.image(img, caption=f"Scatterplot di {var_1, var_2}")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=data_tot, 
                        x=var_1, 
                        y=var_2, 
                        hue='color', 
                        palette=['purple', 'lightgray'], 
                        alpha=0.6
                        )
        fig.tight_layout()
        fig.savefig(scatt_img_path)
        st.pyplot(fig)     

    st.markdown("---")

with tab3:
    st.title('Feature & Target')

    # DISTRIBUZIONE QUALITY
    st.subheader(f"Distribuzione dei punteggi di qualità del vino")
    frequenze = data_tot['quality'].value_counts().reset_index()
    frequenze.columns = ['Quality', 'Frequenza']

    data_tot['quality_cat'] = pd.Categorical(data_tot['quality'], categories=list(range(0, 11)), ordered=True)

    # Percorso dell'immagine salvata
    target_img_path = f"images/target_dist.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(target_img_path):
        img = Image.open(target_img_path)
        st.image(img, caption=f"Distribuzione dei punteggi di qualità del vino (range 0–10)")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=data_tot, 
                      x ='quality_cat', 
                      hue='color', 
                      palette=['purple', 'lightgray'], 
                      alpha=0.6, 
                      ax=ax
                      )
        ax.set_xlabel('Punteggio di Qualità')
        ax.set_ylabel('Numero di Campioni')
        plt.tight_layout()
        fig.savefig(target_img_path)
        st.pyplot(fig)

    st.markdown("""### Obiettivo
Collegandosi alle analisi precedenti; analizziamo la distribuzione della qualità per tipo di vino.
Lo scopo è individuare se un colore di vino ha ottenuto punteggi significativamente più alti o più bassi dell’altro; ossia scoprire se il colore del vino influisce sul punteggio ad esso assegnato.




### Osservazioni

- Nei punteggi fortemente positivi **≥ 7**, i vini bianchi sono in proporzione nettamente più presenti dei vini rossi, anche tenendo in considerazione il fatto che nel dataset i vini bianchi sono circa il 75% del totale.
- Per quel che riguarda i punteggi medi **5 e 6**, possiamo osservare come per la valutazione sufficienze la proporzione tra vini bianchi e rossi è a favore dei primi, mentre per il punteggio insufficiente i vini rossi restano quasi costanti in numero mentre quelli bianchi calano di circa 600 unità; facendo sì che ora i vini rossi siano in proporzione più di quelli bianchi rispetto alla distribuzione globale del dataset.
- Nei voti gravemente insufficienti **3 e 4**,possiamo notare per il voto 4 una proporzione dei colori simile a quella originale del dataset, mentre per il voto 3 i vini bianchi sono in proporzione di più, ma su un numero basso di osservazioni.

### Conclusione

Dal grafico emerge un chiaro pattern: i punteggi alti sono più presenti in proporzione tra i vini bianchi, mentre un voto insufficiente mostra una distribuzione del colore del vino con più vini rossi di quanti ce ne sono in proporzione in tutto il dataset.
""")
    st.markdown("---")

    col1, col2 = st.columns(2)
    var_3 = col1.selectbox("Select a variable", covariate, key="selectbox_var_3")
    st.subheader(f"Distribuzione di {var_3} per quality")
    st.markdown("Questo barplot permette di analizzare la distribuzione della variabile in modo stratificato: sull'asse delle ascisse i dati sono suddivisi per livello di qualità, mentre il colore delle barre distingue tra vino rosso e vino bianco.")
    # Percorso dell'immagine salvata
    dist_img_path = f"images/dist_{var_3.replace(' ', '_')}.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(dist_img_path):
        img = Image.open(dist_img_path)
        st.image(img, caption=f"Distribuzione di {var_3} per quality")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=data_tot, 
                    x = "quality", 
                    y = var_3, 
                    hue='color', 
                    palette=['purple', 'lightgray'], 
                    alpha=0.6
                    )
        fig.tight_layout()
        fig.savefig(dist_img_path)
        st.pyplot(fig)
    
    st.markdown("---")

with tab4:
    st.title('Analisi binaria')

    # Creazione di una colonna binaria per la qualità [la usiamo negli scatterplot]
    data_tot['quality_group'] = data_tot['quality'].apply(lambda q: 'high' if q > 5 else 'low')

    col1, col2 = st.columns(2)
    var_5 = col1.selectbox('Seleziona una variabile', covariate, key="selectbox_var_5")
    st.subheader(f'Distribuzione di {var_5}')
    st.markdown('Questo istogramma mostra la distribuzione di una feature stratificata per livello di qualità (bassa vs. alta), permettendo di confrontare la frequenza dei valori tra i due gruppi.')
    # Percorso dell'immagine salvata
    dist_targ2_img_path = f"images/dist_targ2_{var_5.replace(' ', '_')}.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(dist_targ2_img_path):
        img = Image.open(dist_targ2_img_path)
        st.image(img, caption=f"Distribuzione di {var_5}")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(data_tot, 
                     x=var_5, 
                     hue="quality_group", 
                     kde=True, 
                     bins=30, 
                     palette={'low': 'red', 'high': 'green'}, 
                     alpha=0.5
                     )
        fig.tight_layout()
        fig.savefig(dist_targ2_img_path)
        st.pyplot(fig)
        # GRAFICI CORRELAZIONE > 0.6
    st.subheader("Correlation plot")
    st.markdown('Si riportano le coppie di variabili che hanno ottenuto una correlazione maggiore di |0.6|, vengono rappresentate tramite uno scatterplot stratificato in base alla qualità del vino (alta vs bassa)')

    # Calcolo della matrice di correlazione
    correlation_matrix = data_tot.corr(numeric_only=True)

    # Identifica le coppie con correlazione forte (>|0.6| e != 1)
    threshold = 0.6
    correlated_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)): #così vediamo solo le coppie sopra la diagonale 
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                correlated_pairs.append((col1, col2, corr_value))
    
    # Percorso dove salvare il grafico
    os.makedirs("images", exist_ok=True)
    corr_scat_img_path = "images/corr_scat_grid.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(corr_scat_img_path):
        img = Image.open(corr_scat_img_path)
        st.image(img, caption="Scatterplot delle variabili fortemente correlate")

    # Altrimenti genero la griglia di scatterplot e la salvo
    else:
        n = len(correlated_pairs)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()
        
        for idx, (col1, col2, corr_value) in enumerate(correlated_pairs):
            sns.scatterplot(
                data=data_tot,
                x=col1,
                y=col2,
                hue='quality_group',
                alpha=0.6,
                ax=axes[idx],
                palette={'low': 'red', 'high': 'green'}
                )
            axes[idx].set_title(f'{col1} vs {col2}\ncorr = {corr_value:.2f}')
            axes[idx].legend(loc='best', fontsize='small')

        # Nasconde assi inutilizzati
        for j in range(len(correlated_pairs), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.savefig(corr_scat_img_path)
        st.pyplot(fig)

    st.markdown("---")
    



    # GRAFICI A VIOLINO
    st.subheader("Violin plot")
    st.markdown("""Per ogni feature numerica è stato creato un **violinplot** con:
  - x =  quality,
  - y =  feature""")
    st.markdown('In generale, questi plot sono molto utili per farsi un’idea visiva di quali variabili chimiche potrebbero influenzare di più la qualità percepita del vino.')

    # Lista delle feature numeriche (escludendo 'quality')
    features = data_tot.select_dtypes(include='number').columns.drop('quality')

    # Percorso immagine
    os.makedirs("images", exist_ok=True)
    violin2_img_path = "images/violin_plot_grid.png"

    # Se il file esiste, lo carico direttamente
    if os.path.exists(violin2_img_path):
        img = Image.open(violin2_img_path)
        st.image(img, caption="Violin plot per ogni feature numerica")

    # Altrimenti genero e salvo il grafico
    else:
        n = len(features)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for idx, feature in enumerate(features):
            sns.violinplot(
                data=data_tot,
                y=feature,
                hue='quality_group',
                split=True,
                palette={'low': 'red', 'high': 'green'},
                ax=axes[idx],
                alpha=0.6
            )
            axes[idx].set_title(f'{feature} vs Quality (split by group)')
            axes[idx].set_xlabel('Quality')
            axes[idx].set_ylabel(feature)

        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.savefig(violin2_img_path)
        st.pyplot(fig)
    

    st.markdown("---")

with tab5:
    st.title('Analisi a 3 categorie')

    # Creazione di una colonna categorica per la qualità [la usiamo negli scatterplot]
    data_tot['quality_group'] = data_tot['quality'].apply(
        lambda q: 'high' if q in [7, 8, 9] else ('medium' if q in [5, 6] else 'low')
        )

    col1, col2 = st.columns(2)
    var_6 = col1.selectbox('Seleziona una variabile', covariate)
    st.subheader(f'Distribuzione di {var_6}')
    st.markdown('Questo istogramma mostra la distribuzione di una feature stratificata per livello di qualità (bassa vs media vs alta), permettendo di confrontare la frequenza dei valori tra i due gruppi. ')
    # Percorso dell'immagine salvata
    dist_targ3_img_path = f"images/dist_targ3_{var_6.replace(' ', '_')}.png"

    # Se l'immagine esiste, la carico direttamente
    if os.path.exists(dist_targ3_img_path):
        img = Image.open(dist_targ3_img_path)
        st.image(img, caption=f"Distribuzione di {var_6}")

    # Altrimenti genero il grafico, lo salvo, e lo mostro
    else:
        fig, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(data_tot, 
                     x=var_6, 
                     hue="quality_group", 
                     kde=True, 
                     bins=30, 
                     palette={'low': 'red', 'medium': 'orange', 'high': 'green'}, 
                     alpha=0.6
                     )
        fig.tight_layout()
        fig.savefig(dist_targ3_img_path)
        st.pyplot(fig)

    st.markdown("---")

    # GRAFICI CORRELAZIONE > 0.6
    st.subheader("Correlation plot")
    st.markdown('Si riportano le coppie di variabili che hanno ottenuto una correlazione maggiore di |0.6|, vengono rappresentate tramite uno scatterplot stratificato in base alla qualità del vino (alta vs media vs bassa)')

    # Checkbox per selezionare categorie
    st.markdown("Seleziona le categorie da visualizzare:")
    col1, col2, col3 = st.columns(3)
    low = col1.checkbox("Low", value=True)
    medium = col2.checkbox("Medium", value=True)
    high = col3.checkbox("High", value=True)

    # Costruisci la lista delle categorie selezionate
    selected_groups = []
    if low:
        selected_groups.append("low")
    if medium:
        selected_groups.append("medium")
    if high:
        selected_groups.append("high")

    
    # Filtra i dati in base alla selezione
    filtered_data = data_tot[data_tot['quality_group'].isin(selected_groups)]
    
    # Calcolo della matrice di correlazione
    correlation_matrix = data_tot.corr(numeric_only=True)
    
    # Identifica le coppie con correlazione forte (>|0.6| e != 1)
    threshold = 0.6
    correlated_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)): #così vediamo solo le coppie sopra la diagonale 
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                correlated_pairs.append((col1, col2, corr_value))

    # Genera nome file dinamico in base alle categorie selezionate
    selected_str = "_".join(sorted(selected_groups))
    os.makedirs("images", exist_ok=True)
    corr_scat3_img_path = f"images/correlated_scatter_grid_{selected_str}.png"

    # Carica immagine se già esiste
    if os.path.exists(corr_scat3_img_path):
        img = Image.open(corr_scat3_img_path)
        st.image(img, caption=f"Scatterplot correlazioni forti – gruppi: {', '.join(selected_groups)}")

    # Altrimenti genera e salva la griglia
    else:
        n = len(correlated_pairs)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for idx, (col1_name, col2_name, corr_value) in enumerate(correlated_pairs):
            sns.scatterplot(
                data=filtered_data,
                x=col1_name,
                y=col2_name,
                hue='quality_group',
                alpha=0.6,
                ax=axes[idx],
                palette={'low': 'red', 'medium': 'orange', 'high': 'green'}
            )
            axes[idx].set_title(f'{col1_name} vs {col2_name}\ncorr = {corr_value:.2f}')
            axes[idx].legend(loc='best', fontsize='small')

        # Rimuove assi vuoti
        for j in range(len(correlated_pairs), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.savefig(corr_scat3_img_path)
        st.pyplot(fig)

    st.markdown("---")


    # GRAFICI A VIOLINO
    st.subheader("Violin plot")
    st.markdown("""Per ogni feature numerica è stato creato un **violin plot** con:
  - x =  wine type ,
  - y =  feature""")

    st.markdown('In generale, questi plot sono molto utili per farsi un’idea visiva di quali variabili chimiche potrebbero influenzare di più la qualità percepita del vino.')

    # Lista delle feature numeriche (escludendo 'quality')
    features = data_tot.select_dtypes(include='number').columns.drop('quality')

    # Percorso immagine
    os.makedirs("images", exist_ok=True)
    violin3_img_path = "images/violin_plot_grid2.png"

    # Se il file esiste, lo carico direttamente
    if os.path.exists(violin3_img_path):
        img = Image.open(violin3_img_path)
        st.image(img, caption="Violin plot per ogni feature numerica")

    # Altrimenti genero e salvo il grafico
    else:
        n = len(features)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for idx, feature in enumerate(features):
            sns.violinplot(
                data=data_tot,
                y=feature,
                hue='quality_group',
                split=False,
                palette={'low': 'red','medium': 'orange', 'high': 'green'},
                ax=axes[idx],
                alpha=0.6
                )
            axes[idx].set_title(f'{feature} vs Quality (split by group)')
            axes[idx].set_xlabel('Quality')
            axes[idx].set_ylabel(feature)

        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.savefig(violin3_img_path)
        st.pyplot(fig)

    st.markdown("---")

with tab6:
    st.title('Analisi della correlazione')
    st.markdown("""### Risultati chiave osservati:
Abbiamo osservato le coppie di variabili con la maggior correlazione sia positiva sia negativa, per vedere se si presentavano sovrapposizioni evidenti tra punti di colore diverso o se invece vini di colore diverso presentavano rette con coefficiente di correlazione simile ma spostate nel piano. È possibile vedere come nel plot density vs alcohol i vini rossi siano leggermente spostati a destra sull’asse density rispetto ai vini bianchi, ma non si registrano spostamenti lungo l’asse di alcohol, a indicare come la correlazione tra le due variabili si modifichi leggermente tra i due tipi di vino, anche se non in maniera significativa.

""")
    img = Image.open(f"images/corr_scat_col.png")

    st.image(img, caption=f"Scatterplot suddiviso in base al colore del vino")
    st.subheader("Violin plot")
    st.markdown("""Per ogni feature numerica è stato creato un **violin plot** con:
  - x =  wine type ,
  - y =  feature""")

    img = Image.open(f"images/violin_plot_colore.png")
    st.image(img, caption=f"Violin plot suddiviso in base al colore del vino")
    st.markdown("""

### Osservazioni principali
- **alcohol**: sembra avere distribuzione molto simile tra entrambi i tipi di vino.
- **density**: la moda della distribuzione dei vini bianchi è circa 0,99; valore invece pressochè assente nella distribuzione dei vini rossi.
- **total_sulfur_dioxide**: variabile fortemente discriminante. Mentre i vini rossi hanno valori bassi e tendenti allo zero, i vini bianche mostrano una distribuzione più variabile e con valori sensibilmente più ampi.
- **residual_sugar**: mentre nei vini rossi l’intera distribuzione è concentrata attorno a pochi valori, nei vini bianchi osserviamo un picco sugli stessi valori ma anche una distribuzione più ampia che comprende valori più alti.
- **volatile acidity** e **fixed acidity**: i vini rossi registrano in media valori più alti negli indici di acidità. I vini bianchi presentano distribuzioni regolari e (quasi)normali con in media valori di acidità bassi.


### Conclusione
Alcune variabili sembrano avere un forte potere discriminante nella tipologia di vino, andando dunque a catturare le differenze strutturali nei componenti chimici tra i vini rossi e i vini bianchi.
""")
    st.header('PCA')
    img = Image.open(f"images/PCA.png")
    st.image(img, caption=f"PCA")
    st.markdown("""###  PCA – Principal Component Analysis
- Metodo lineare di riduzione dimensionale, che proietta i dati lungo assi ortogonali che massimizzano la varianza globale.
- Le prime due componenti spiegano il **47.43% della varianza totale** del dataset.
- Le componenti principali sembrano poter creare una netta separazione tra i due vini di diverso colore. In particolare, i vini che presentano sulla prima componente un valore minore di -1,25 circa sono quasi tutti vini rossi; viceversa quasi tutti i vini con punteggio maggiore di -1,25 son bianchi. Guardando al contributo delle variabili più importanti per ogni componente, possiamo dire che i vini rossi presentano maggiore acidità e maggior numero di cloruri; mentre i vini bianchi presentano maggior quantità di diossido di zolfo e di zuccheri non fermentati.
La tecnica esplorativa utilizzata dunque **evidenzia una separazione netta tra le due tipologie di vino analizzate**.  
""")
   

