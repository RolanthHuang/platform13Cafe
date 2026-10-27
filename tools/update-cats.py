#!/usr/bin/env python3
"""Copy cats/cat-kit.js into index.html (between the P13 CAT KIT markers).

Usage (from the repo root):  python3 tools/update-cats.py
"""
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
html_path = root / 'index.html'
html = html_path.read_text(encoding='utf-8')
kit = (root / 'cats' / 'cat-kit.js').read_text(encoding='utf-8')
kit = kit.replace('export function createCatKit(', 'function createCatKit(', 1)
assert 'export ' not in kit, 'cat-kit.js may only export createCatKit'
assert '</script' not in kit, 'cat-kit.js must not contain a closing script tag'

BEGIN = '/* ==== P13 CAT KIT BEGIN (source: cats/cat-kit.js) ==== */\n'
END = '\n/* ==== P13 CAT KIT END ==== */'
i = html.index(BEGIN) + len(BEGIN)
j = html.index(END)
html_path.write_text(html[:i] + kit + html[j:], encoding='utf-8')
print(f'updated {html_path.name} ({len(kit)} chars of cat kit)')
