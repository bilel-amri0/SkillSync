"""
ULTIMATE FIX - Directly replace the problematic section
"""

# Read the entire file
with open('main_simple_for_frontend.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")
print(f"Line 680 before fix: {lines[679].strip()}")

# Find the problematic try block and remove it
# Look for "return formatted_recommendations" followed by empty line and "try:"
fixed_lines = []
skip_until = -1

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Skip lines in the problematic block (680-692 approximately)
    if skip_until > 0 and line_num <= skip_until:
        continue
    
    # Detect the start of problematic block
    if line_num == 680 and line.strip() == 'try:':
        # Skip this try block and everything until we find the next major section
        # Look ahead to find where to resume
        for j in range(i, min(i + 50, len(lines))):
            if 'import auth' in lines[j] or 'AUTH_ENABLED' in lines[j] or 'app = FastAPI' in lines[j]:
                skip_until = j - 1
                print(f"Skipping lines {line_num} to {skip_until}")
                break
        continue
    
    fixed_lines.append(line)

# Write the fixed content
with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"✅ Fixed! New total lines: {len(fixed_lines)}")
print(f"✅ Removed {len(lines) - len(fixed_lines)} problematic lines")
