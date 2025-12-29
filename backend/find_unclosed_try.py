import re

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== ALL TRY STATEMENTS BEFORE LINE 667 ===")
for i, line in enumerate(lines[:667], 1):
    stripped = line.strip()
    indent = len(line) - len(line.lstrip())
    if stripped.startswith('try:'):
        print(f'Line {i} (indent {indent}): {stripped}')

print("\n=== CHECKING FOR UNCLOSED TRY BLOCKS ===")

# Simple tracking
try_stack = []

for i, line in enumerate(lines[:667], 1):
    stripped = line.strip()
    indent = len(line) - len(line.lstrip())
    
    if stripped.startswith('try:'):
        try_stack.append((i, indent, stripped))
    elif stripped.startswith(('except', 'finally:')):
        if try_stack:
            # Find matching try (same or higher indent level)
            for j in range(len(try_stack)-1, -1, -1):
                if try_stack[j][1] == indent:
                    try_stack.pop(j)
                    break

print("UNCLOSED TRY BLOCKS:")
for t in try_stack:
    print(f"  Line {t[0]}: {t[2]}")
    # Show context
    start = max(0, t[0]-2)
    end = min(len(lines), t[0]+10)
    print(f"  Context (lines {start+1}-{end}):")
    for k in range(start, end):
        marker = ">>>" if k == t[0]-1 else "   "
        print(f"    {marker} {k+1}: {lines[k].rstrip()}")
    print()
