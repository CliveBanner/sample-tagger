import json
import re

def classify_cluster(cluster):
    paths = cluster.get('paths', [])
    model_label = cluster.get('model_label', '')
    agreement = cluster.get('agreement', 0)
    c_id = cluster['id']
    
    path_text = " ".join(paths).lower()
    
    # helper functions
    def has_any(keywords):
        return any(k in path_text for k in keywords)
        
    def has_regex(pattern):
        return bool(re.search(pattern, path_text))
        
    # Rules check
    
    # ETHNIC
    if 'ethnic/' in path_text:
        if has_any(['jews harp']): return {'id': c_id, 'label': 'jawharp', 'confidence': 'high', 'note': 'ETHNIC/Jews Harp'}
        if has_any(['didgeridoo']): return {'id': c_id, 'label': 'didgeridoo', 'confidence': 'high', 'note': 'ETHNIC/Didgeridoo'}
        if has_any(['koto', 'sitar', 'mandolin', 'banjo']): return {'id': c_id, 'label': 'pluck', 'confidence': 'high', 'note': 'ETHNIC plucked'}
        if has_any(['hmrd dulcimer', 'glockenspiel']): return {'id': c_id, 'label': 'mallet', 'confidence': 'high', 'note': 'ETHNIC mallet'}
        if has_any(['accordian']): return {'id': c_id, 'label': 'winds', 'confidence': 'high', 'note': 'ETHNIC winds'}
        if has_any(['tabla', 'african drums', 'world whistles', 'perc']): return {'id': c_id, 'label': 'perc', 'confidence': 'high', 'note': 'ETHNIC perc'}
        
    # Foley
    if has_any(['foley', 'paper tearing', 'eating', 'cicada', 'cricket', 'ambient', 'ambience']):
        return {'id': c_id, 'label': 'foley', 'confidence': 'high', 'note': 'foley/ambience'}

    # Dialog (Game voice lines)
    if has_regex(r'\b(vo/|npc/|barney/|metropolice/|odessa/|ravenholm/|streetwar/|coast/cardock/|alyx_)\b'):
        return {'id': c_id, 'label': 'dialog', 'confidence': 'high', 'note': 'Game voice lines'}
        
    # Strings / Pizzicato
    if has_regex(r'\b(piz|pizm|pizf|pibf|vi|cl|sus|slowv|isusv|violin|cello|strings|viola)\b'):
        return {'id': c_id, 'label': 'strings', 'confidence': 'high', 'note': 'String identifiers found'}
        
    # Guitar
    if has_regex(r'\b(guitar|gtr|acoustic guitar|electric guitar)\b'):
        return {'id': c_id, 'label': 'guitar', 'confidence': 'high', 'note': 'Explicit guitar path'}
        
    # Piano
    if has_regex(r'\b(piano|grand|steinway|upright piano)\b'):
        return {'id': c_id, 'label': 'piano', 'confidence': 'high', 'note': 'Explicit piano path'}
        
    # Organ
    if has_regex(r'\b(organ|hammond|b3|vox continental|pipe organ|mellotron)\b'):
        return {'id': c_id, 'label': 'organ', 'confidence': 'high', 'note': 'Explicit organ path'}
        
    # Keys
    if has_regex(r'\b(keys|rhodes|wurlitzer|dx7|cp-70|clavinet)\b'):
        return {'id': c_id, 'label': 'keys', 'confidence': 'high', 'note': 'Explicit keys path'}
        
    # Brass
    if has_regex(r'\b(brass|trumpet|trp|french horn|trombone|tuba)\b') or 'lf' in path_text.split('/'):
        return {'id': c_id, 'label': 'brass', 'confidence': 'high', 'note': 'Explicit brass path'}
        
    # Winds
    if has_regex(r'\b(winds|flute|clarinet|saxophone|oboe|accordion|harmonica)\b'):
        return {'id': c_id, 'label': 'winds', 'confidence': 'high', 'note': 'Explicit winds path'}
        
    # Vocal
    if has_regex(r'\b(vocal|choir|acapella|singing|laughter)\b'):
        return {'id': c_id, 'label': 'vocal', 'confidence': 'high', 'note': 'Explicit vocal path'}
        
    # FX
    if has_regex(r'\b(fx|laser|explosion|impact|sweep)\b'):
        return {'id': c_id, 'label': 'fx', 'confidence': 'high', 'note': 'Explicit fx path'}

    # Paths beat the model
    # Kicks
    if has_regex(r'\b(bd|kick|bassdrum|909bd)\b') or 'tx-81z bd' in path_text or 'ielectribe-kick' in path_text or 'mbase' in path_text:
        return {'id': c_id, 'label': 'kick', 'confidence': 'high', 'note': 'Explicit kick path'}
        
    # Snares
    if has_regex(r'\b(sn|sd|snare|snr)\b'):
        return {'id': c_id, 'label': 'snare', 'confidence': 'high', 'note': 'Explicit snare path'}
        
    # Hi-hats
    if has_regex(r'\b(hh|hat|closedhat|openhat|hi-hat|hihat)\b'):
        return {'id': c_id, 'label': 'hihat', 'confidence': 'high', 'note': 'Explicit hihat path'}
        
    # Claps
    if has_regex(r'\b(clap|handclap)\b'):
        return {'id': c_id, 'label': 'clap', 'confidence': 'high', 'note': 'Explicit clap path'}
        
    # Toms
    if has_regex(r'\b(tom|tm|ptm|drtm|tomtom)\b'):
        return {'id': c_id, 'label': 'tom', 'confidence': 'high', 'note': 'Explicit tom path'}
        
    # Cymbals
    if has_regex(r'\b(cymbal|crash|ride|splash)\b'):
        return {'id': c_id, 'label': 'cymbal', 'confidence': 'high', 'note': 'Explicit cymbal path'}
        
    # Bass
    if has_regex(r'\b(bass)\b') and 'drum' not in path_text:
        return {'id': c_id, 'label': 'bass', 'confidence': 'high', 'note': 'Explicit bass path'}
        
    # Loops
    if has_regex(r'\b(bpm|loop)\b') and not has_regex(r'\b(one shot|oneshot)\b'):
        if 'drum' in path_text:
            return {'id': c_id, 'label': 'drums', 'confidence': 'high', 'note': 'Drum loop'}
        if has_regex(r'\b(bass)\b'):
            return {'id': c_id, 'label': 'bass', 'confidence': 'high', 'note': 'Bass loop'}
        return {'id': c_id, 'label': 'synth', 'confidence': 'med', 'note': 'Loop, defaulting to synth/mixed'}


    # Opaque AMG AKAI hits
    # E.g. 065SLOW9R
    if has_regex(r'\b\d{3}[A-Z0-9]+\b') and len(paths) > 0 and 'amg' in path_text:
        if model_label in ['drumhit', 'kick', 'snare', 'clap'] and agreement >= 0.80:
            return {'id': c_id, 'label': 'drumhit', 'confidence': 'med', 'note': 'AMG opaque hit, relying on model'}
            
    # Fallbacks based on agreement
    if agreement >= 0.85:
        return {'id': c_id, 'label': model_label, 'confidence': 'high' if agreement > 0.95 else 'med', 'note': f'No explicit path, model={model_label} (agreement {agreement})'}
        
    if 0.60 <= agreement < 0.85:
        return {'id': c_id, 'label': model_label, 'confidence': 'med', 'note': f'No explicit path, moderate agreement {agreement}'}
        
    # Default skip
    return {'id': c_id, 'label': None, 'confidence': 'low', 'note': f'Low agreement {agreement} and opaque paths'}

def main():
    try:
        with open('/tmp/cluster_dump.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading dump: {e}")
        return
        
    results = []
    for cluster in data:
        results.append(classify_cluster(cluster))
        
    with open('/tmp/cluster_labels_batch1.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Processed {len(results)} clusters.")

if __name__ == '__main__':
    main()
