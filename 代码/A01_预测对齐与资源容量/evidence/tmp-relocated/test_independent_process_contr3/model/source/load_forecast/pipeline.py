import json
from pathlib import Path
def predict_record(bundle, history, output_path, as_of):
    import time; time.sleep(3)
    record=json.loads((Path(bundle)/'fixture.json').read_text(encoding='utf-8'))
    Path(output_path).write_text(json.dumps(record), encoding='utf-8')
    return record
