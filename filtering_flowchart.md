# Feedback Filtering Flowchart

## 🔄 Simple Decision Flow

```
START: Raw Message
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TEXT PREPROCESSING                                    │
│                                                                                     │
│  ┌─ Clean text (lowercase, normalize)                                             │
│  ├─ Extract keywords from 50+ feedback terms                                     │
│  ├─ Count regex pattern matches (10 patterns)                                     │
│  └─ Calculate weighted score                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION POINT 1                                     │
│                                                                                     │
│  Does message contain core feedback terms?                                         │
│  (feedback, review, rating, opinion, suggestion)                                  │
│                                                                                     │
│  ┌─ YES ──────────────────────────────────► POTENTIAL FEEDBACK                     │
│  └─ NO ──────────────────────────────────► Continue to next check                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION POINT 2                                     │
│                                                                                     │
│  Are there 2+ feedback keywords?                                                  │
│                                                                                     │
│  ┌─ YES ──────────────────────────────────► POTENTIAL FEEDBACK                     │
│  └─ NO ──────────────────────────────────► Continue to next check                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION POINT 3                                     │
│                                                                                     │
│  Are there 3+ pattern matches?                                                    │
│                                                                                     │
│  ┌─ YES ──────────────────────────────────► POTENTIAL FEEDBACK                     │
│  └─ NO ──────────────────────────────────► Continue to next check                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION POINT 4                                     │
│                                                                                     │
│  Is weighted score >= 3.0?                                                        │
│                                                                                     │
│  ┌─ YES ──────────────────────────────────► POTENTIAL FEEDBACK                     │
│  └─ NO ──────────────────────────────────► NON-FEEDBACK                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              NEXT STEPS                                           │
│                                                                                     │
│  POTENTIAL FEEDBACK ──► Check cache ──► LLM analysis (if needed)                  │
│  NON-FEEDBACK ───────► Mark as non-feedback (no LLM call)                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Filtering Statistics

### **Typical Results for 1000 Messages:**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FILTERING BREAKDOWN                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

📥 INPUT: 1000 messages
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING FILTER                                    │
│                                                                                     │
│  🔍 Potential Feedback: 250 messages (25%)                                        │
│  ❌ Non-Feedback: 750 messages (75%)                                              │
│                                                                                     │
│  💰 Cost Savings: 75% reduction in LLM calls                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CACHE FILTER                                         │
│                                                                                     │
│  🔄 Cached Results: 100 messages (40% of potential)                               │
│  🤖 New LLM Calls: 150 messages (60% of potential)                               │
│                                                                                     │
│  💰 Additional Savings: 40% reduction in new API calls                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL RESULTS                                        │
│                                                                                     │
│  ✅ Confirmed Feedback: 180 messages (18%)                                        │
│  ❌ No Feedback: 820 messages (82%)                                               │
│                                                                                     │
│  💰 Total Cost Reduction: 85% vs analyzing all messages                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Benefits

1. **Cost Efficiency**: 75-90% reduction in API costs
2. **Speed**: Only 10-25% of messages need LLM analysis
3. **Accuracy**: Preprocessing catches obvious cases
4. **Scalability**: Handles large datasets efficiently
5. **Caching**: Avoids redundant analysis of similar messages

## 🔧 Customization Options

You can adjust the filtering sensitivity by modifying:

- **Keyword threshold**: `len(keywords) >= 2` → `len(keywords) >= 1`
- **Pattern threshold**: `pattern_matches >= 3` → `pattern_matches >= 2`
- **Score threshold**: `score >= 3.0` → `score >= 2.0`
- **Core terms**: Add/remove priority terms
- **Keywords**: Expand/reduce keyword list
- **Patterns**: Modify regex patterns 