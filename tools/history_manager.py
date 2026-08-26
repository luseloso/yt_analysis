import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'run_history.json')

def load_run_history() -> List[Dict[str, Any]]:
    """Loads all past runs from run_history.json sorted by timestamp descending."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f'Error loading history: {e}')
    return []

def save_run_to_history(
    video_url: str,
    video_id: str,
    template: str,
    model: str,
    duration_seconds: Optional[int],
    elapsed_seconds: float,
    report_markdown: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Saves a new run entry to run_history.json."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history = load_run_history()
    
    run_entry = {
        'id': f'run_{int(time.time())}_{video_id or "custom"}',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_url': video_url,
        'video_id': video_id,
        'template': template,
        'model': model,
        'duration_seconds': duration_seconds,
        'elapsed_seconds': round(elapsed_seconds, 1),
        'report_markdown': report_markdown,
        'output_file': output_file
    }
    
    # Prepend newest first, keep last 50 runs
    history.insert(0, run_entry)
    history = history[:50]
    
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'Error saving history: {e}')
        
    return run_entry

def get_run_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a specific run by its unique ID."""
    history = load_run_history()
    for item in history:
        if item.get('id') == run_id:
            return item
    return None

def clear_run_history():
    """Clears all saved history."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            print(f'Error clearing history: {e}')
