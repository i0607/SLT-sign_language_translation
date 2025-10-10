def build_vocabularies(data):
    """
    Build gloss and word vocabularies from training data
    
    Args:
        data: pandas DataFrame with 'orth_english' and 'translation_english' columns
    
    Returns:
        gloss_vocab: dict mapping glosses to IDs
        word_vocab: dict mapping words to IDs
    """
    all_glosses = []
    all_words = []
    
    # Collect all glosses and words
    for i in range(len(data)):
        glosses = str(data.iloc[i]['orth_english']).split()
        words = str(data.iloc[i]['translation_english']).split()
        all_glosses.extend(glosses)
        all_words.extend(words)
    
    # Create vocabularies
    unique_glosses = sorted(list(set(all_glosses)))
    unique_words = sorted(list(set(all_words)))
    
    # Gloss vocabulary (for recognition)
    gloss_vocab = {'<BLANK>': 0}  # CTC blank token at index 0
    for i, gloss in enumerate(unique_glosses):
        gloss_vocab[gloss] = i + 1
    
    # Word vocabulary (for translation)
    word_vocab = {
        '<SOS>': 0,   # Start of sentence
        '<EOS>': 1,   # End of sentence
        '<UNK>': 2    # Unknown word
    }
    for i, word in enumerate(unique_words):
        word_vocab[word] = i + 3
    
    return gloss_vocab, word_vocab