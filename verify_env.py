import platform
import os

try:
    from pyzbar import pyzbar
    print("pyzbar: OK")
except Exception as e:
    print(f"pyzbar: FAIL -> {e}")

try:
    #import pytesseract
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\anhng\miniconda3\envs\blood_bag_ocr\Library\bin\tesseract.exe'
    import pytesseract 

    if platform.system() == "Windows":
        _tess = os.path.join(
            os.environ.get("CONDA_PREFIX", ""),
            "Library", "bin", "tesseract.exe"
        )
        if os.path.exists(_tess):
            pytesseract.pytesseract.tesseract_cmd = _tess
    print(f"Tesseract: {pytesseract.get_tesseract_version()}")
    langs = pytesseract.get_languages(config="")
    print(f"Languages: {langs}")
    print("German present:", "deu" in langs)
except Exception as e:
    print(f"pytesseract: FAIL -> {e}")

# lxml
try:
    from lxml import etree
    print("lxml: OK")
except Exception as e:
    print(f"lxml: FAIL -> {e}")

# editdistance
try:
    import editdistance
    assert editdistance.eval("hello", "helo") == 1
    print("editdistance: OK")
except Exception as e:
    print(f"editdistance: FAIL -> {e}")

# Pillow
try:
    from PIL import Image
    print(f"Pillow: OK")
except Exception as e:
    print(f"Pillow: FAIL -> {e}")


