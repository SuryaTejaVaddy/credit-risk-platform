import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('data/processed', exist_ok=True)

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=== FinBERT Sentiment Analysis ===')

df = pd.read_csv('data/raw/financial_sentiment.csv')
print(f'Loaded {len(df)} sentences')

print('Loading FinBERT model (ProsusAI/finbert)...')
print('Note: First run downloads ~500MB — subsequent runs use cache.')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model_hf  = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    nlp       = pipeline('sentiment-analysis', model=model_hf, tokenizer=tokenizer,
                          truncation=True, max_length=512)

    print('Running inference...')
    results = []
    batch_size = 32
    sentences  = df['sentence'].tolist()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        out   = nlp(batch)
        results.extend(out)
        if (i // batch_size) % 5 == 0:
            print(f'  Processed {min(i+batch_size, len(sentences))}/{len(sentences)}')

    label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['finbert_label'] = [r['label'].lower() for r in results]
    df['finbert_score'] = [r['score'] for r in results]
    df['sentiment_num'] = df['finbert_label'].map(label_map)

except Exception as e:
    print(f'FinBERT inference failed ({e})')
    print('Falling back to rule-based sentiment scoring...')
    positive_words = ['strong','surged','exceeded','improved','high','profit','growth','gain']
    negative_words = ['declined','challenges','debt','layoffs','cut','loss','risk','decline']
    def rule_sentiment(text):
        t = text.lower()
        pos = sum(w in t for w in positive_words)
        neg = sum(w in t for w in negative_words)
        if pos > neg:   return 'positive', 0.80
        elif neg > pos: return 'negative', 0.80
        else:           return 'neutral',  0.70
    labels_scores = df['sentence'].apply(rule_sentiment)
    df['finbert_label'] = [x[0] for x in labels_scores]
    df['finbert_score'] = [x[1] for x in labels_scores]
    label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['sentiment_num'] = df['finbert_label'].map(label_map)
    print('Rule-based fallback complete.')

df.to_csv('data/processed/sentiment_scores.csv', index=False)
print(f'\nSaved: data/processed/sentiment_scores.csv')

# ── Distribution plot ──────────────────────────────────────────────────────────
counts = df['finbert_label'].value_counts()
colors = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#2196F3'}
plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values,
               color=[colors.get(l, '#999') for l in counts.index])
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontsize=11)
plt.title('FinBERT Sentiment Distribution on Financial News', fontsize=14)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/processed/sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: data/processed/sentiment_distribution.png')

# ── Confidence histogram ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
for label, color in colors.items():
    sub = df[df['finbert_label'] == label]['finbert_score']
    if len(sub):
        plt.hist(sub, bins=20, alpha=0.6, label=label, color=color)
plt.title('FinBERT Confidence Score Distribution', fontsize=13)
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/sentiment_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: data/processed/sentiment_confidence.png')

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n--- Sentiment Summary ---')
print(df['finbert_label'].value_counts().to_string())
print(f"\nMean confidence: {df['finbert_score'].mean():.3f}")
print(f"Positive rate:   {(df['finbert_label']=='positive').mean():.1%}")
print(f"Negative rate:   {(df['finbert_label']=='negative').mean():.1%}")
print(f"Neutral rate:    {(df['finbert_label']=='neutral').mean():.1%}")

# ── Save sentiment encoder for dashboard ──────────────────────────────────────
meta = {
    'label_counts': df['finbert_label'].value_counts().to_dict(),
    'mean_score':   float(df['finbert_score'].mean()),
}
joblib.dump(meta, 'data/feature_store/sentiment_meta.pkl')
print('Saved: data/feature_store/sentiment_meta.pkl')
print('\n=== Sentiment Analysis Complete ===')
