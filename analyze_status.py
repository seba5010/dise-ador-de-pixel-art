import json
with open('training_status.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

h = d.get('history', [])
valid_losses = [e['g_loss'] for e in h if isinstance(e.get('g_loss'), float)]
min_loss = min(valid_losses)
best_epoch = next(e['epoch'] for e in h if abs(e.get('g_loss', 999) - min_loss) < 0.0001)
print(f'Min g_loss in history: {min_loss} at epoch {best_epoch}')
print(f'Status says best_loss: {d.get("best_loss")}')
print()
print('Epochs 1-10:')
for e in h[:10]:
    print(f'  ep {e["epoch"]}: g_loss={e["g_loss"]}')
print()
print('Past eras:')
for pe in d.get('past_eras', []):
    print(f'  Era {pe.get("era")}: {pe.get("epochs")} epocas, best_loss={pe.get("best_loss")}')
