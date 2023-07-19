# activate environment
conda activate image2image_package

$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver

python zip.py
echo "Zipped files"

conda deactivate image2image_package
