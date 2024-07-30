call conda activate image2image
set cwd=%cd%
call python -X importtime -c "import image2image.qt.dialog_wsireg"
call conda deactivate