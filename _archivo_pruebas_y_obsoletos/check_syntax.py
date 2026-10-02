import py_compile, sys
try:
    py_compile.compile('pixel_ai_engine/train.py', doraise=True)
    print('OK: train.py sintaxis correcta')
except py_compile.PyCompileError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
