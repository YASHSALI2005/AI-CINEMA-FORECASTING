import re

def clean_currency(value):
    print(f"Original: '{value}'")
    if not value or value == 'N/A' or value == '-':
        return 0.0
    # Remove Currency symbols and commas
    clean = re.sub(r'[^\d.]', '', str(value))
    print(f"Cleaned regex: '{clean}'")
    try:
        f = float(clean)
        print(f"Float: {f}")
        return f
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

clean_currency("24.60 cr.")
clean_currency("4 cr.")
clean_currency("1,200.50")
