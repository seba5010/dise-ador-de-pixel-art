import json
from pathlib import Path
from PIL import Image
import numpy as np

root = Path('dataset_frames_individuales')
manifests = list(root.rglob('manifest.json'))
print(f'Total manifests found: {len(manifests)}')

border_issues = []
for m_path in manifests:
    with open(m_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    char = data.get('character', m_path.parent.parent.name)
    variant = data.get('variant_name', data.get('variant', m_path.parent.name))
    frames = data.get('frames', [])
    top_contacts = []
    bottom_contacts = []
    left_contacts = []
    right_contacts = []
    for fr in frames:
        contacts = fr.get('source_border_contacts', [])
        slot = fr.get('slot')
        if 'top' in contacts:
            top_contacts.append(slot)
        if 'bottom' in contacts:
            bottom_contacts.append(slot)
        if 'left' in contacts:
            left_contacts.append(slot)
        if 'right' in contacts:
            right_contacts.append(slot)
            
    if top_contacts or bottom_contacts or left_contacts or right_contacts:
        border_issues.append({
            'char': char,
            'variant': variant,
            'top_count': len(top_contacts),
            'bottom_count': len(bottom_contacts),
            'left_count': len(left_contacts),
            'right_count': len(right_contacts),
            'top_slots': top_contacts,
            'bottom_slots': bottom_contacts,
            'total_frames': len(frames),
            'dir': str(m_path.parent)
        })

border_issues.sort(key=lambda x: (x['bottom_count'] + x['top_count']), reverse=True)
print(f'\nTotal variants with border contacts: {len(border_issues)}\n')
for item in border_issues:
    print(f"=== {item['char'].upper()} - {item['variant']} ===")
    print(f"  Frames afectados: TOP={item['top_count']}, BOTTOM={item['bottom_count']}, LEFT={item['left_count']}, RIGHT={item['right_count']} (de {item['total_frames']} frames)")
    if item['top_count']:
        print(f"  Slots con cabeza cortada (TOP): {item['top_slots']}")
    if item['bottom_count']:
        print(f"  Slots con pies cortados (BOTTOM): {item['bottom_slots']}")
    print()
