
import os

files_to_edit = {
    '/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/sections/results.tex': [
        ('network density---the proportion of possible connections that are realized---increased markedly', 
         'network density (the proportion of possible connections that are realized) increased markedly'),
        ('This asymmetric pattern---where P1\'s threat-dominated structure was not restored despite diplomatic failure---provides',
         'This asymmetric pattern (where P1\'s threat-dominated structure was not restored despite diplomatic failure) provides')
    ],
    '/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/sections/introduction.tex': [
        ('whereas sentiment fully reverts---a pattern we term asymmetric persistence.',
         'whereas sentiment fully reverts (a pattern we term asymmetric persistence).')
    ]
}

for filepath, replacements in files_to_edit.items():
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    for target, replacement in replacements:
        if target in content:
            content = content.replace(target, replacement)
        else:
            print(f"Target not found in {filepath}:\n{target}")
            # Try finding substring
            subtarget = target.split('---')[0]
            if subtarget in content:
                print(f"  Found potential match start: '{subtarget}'")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes made to {filepath}")
