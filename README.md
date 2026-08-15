# Biomarqueurs vocaux de la dépression sur Androids et DAIC-WOZ

Code et calculs d'un stage de six mois co-encadré par le laboratoire Costech (UTC) et le LORIA (Nancy), de février à juillet 2026.

Le travail applique une même chaîne de détection automatique de la dépression à partir de la parole au DAIC-WOZ et à Androids.

### Pipeline

1. Découpage des enregistrements en tours de parole.
2. Extraction des 88 descripteurs eGeMAPS avec openSMILE.
3. Centrage-réduction ajusté sur les seules données d'entraînement.
4. Classification par SVM à noyau gaussien, avec pondération des classes.
5. Agrégation vers un verdict par locuteur, soit en moyennant les descripteurs avant classification (*early fusion*), soit en classant chaque tour puis par vote majoritaire (*late fusion*).

### Résultats

Intra-corpus, F1 macro / UAR / AUC :

| Corpus | Condition | F1 macro | UAR | AUC |
|---|---|---|---|---|
| Androids | lecture | 0,744 | 0,748 | 0,851 |
| Androids | entretien | 0,821 | 0,822 | 0,890 |
| Androids | combiné | 0,766 | 0,769 | 0,871 |
| Androids | entretien, late fusion | 0,816 | 0,825 | 0,893 |
| DAIC-WOZ | early fusion | 0,507 | 0,512 | 0,468 |
| DAIC-WOZ | late fusion | 0,517 | 0,522 | 0,520 |


## Script de segmentation

Le script de segmentation code/chunking_daic.py corrige les défauts propres à ce corpus, à savoir les sessions exclues par la documentation, les bips de synchronisation annotés de cinq façons différentes, les décalages entre transcription et audio sur quatre sessions, les interruptions par un tiers ou un téléphone, les passages anonymisés devenus silencieux, une erreur d'attribution de locuteur et une erreur d'étiquette. Au total 215 segments sont écartés.

## Corpus

Les deux corpus ne sont pas inclus. Il s'agit de :

Gratch et al. (2014), *The Distress Analysis Interview Corpus of human and computer interviews*, LREC.

Tao et al. (2023), *The Androids Corpus: a new publicly available benchmark for speech based depression detection*, Interspeech.

Les chemins attendus sont indiqués dans la cellule de configuration du notebook. Les descripteurs sont mis en cache dans des fichiers `.npz` pour éviter de réextraire l'audio à chaque exécution, et les cellules d'extraction les reconstruisent si les caches sont absents.
