export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset

# generate a small subset to test the training program
python3.8 core/util/preprocessor/argoverse_preprocess_v2.py
