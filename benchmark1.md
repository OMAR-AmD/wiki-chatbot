# Benchmark Jour 1 - Llama2 vs Mistral

## Résultats

| Modèle  | Latence Moyenne | Min   | Max   | Qualité Réponse | Speed   |
|----------|-----------------|-------|-------|-----------------|---------|
| Llama2   | 10.91s          | 8.67s | 13.36s| Bonne           | Moyenne |
| Mistral  | 15.23s          | 8.44s | 27.43s| Moyenne         | Lente   |

## Observations
- **Llama2 :** Temps de réponse plus stable, réponses bien structurées et cohérentes.  
- **Mistral :** Plus variable en latence, parfois trop verbeux ou moins précis dans les explications.  
- Llama2 a montré une meilleure constance dans les trois questions testées.

## Recommandation
**Utiliser Llama2 pour le chatbot wiki.**  
- Raison : Meilleure qualité de réponse et formulation plus claire.  
- La latence (~11s) reste acceptable dans un environnement interne non temps réel.  
- **Priorité :** Qualité et précision > rapidité.
