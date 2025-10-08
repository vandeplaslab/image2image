# activate environment
$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver

python zip.py
echo "Zipped files"
